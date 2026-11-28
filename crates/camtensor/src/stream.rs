//! Per-stream pipeline: RTSP source (async) → gate → decoder → preprocess
//! into every subscribed output.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use camtensor_bitstream::gate::{Gate, GateState};
use camtensor_core::time::now_ns;
use camtensor_core::{
    DecoderPref, EncodedPacket, Error, Frame, FrameMeta, Result, StreamConfig, StreamParams,
};
use camtensor_decode::{Decoder, DecoderOptions};
use camtensor_ingest::{RtspSource, SourceConfig, SourceEvent, SourceHandle};

/// Where packets come from.
pub(crate) enum Source {
    Rtsp(SourceHandle),
    #[cfg(feature = "ffmpeg")]
    File(crate::filesrc::FileSource),
}

impl Source {
    fn stop(self) {
        match self {
            Source::Rtsp(s) => s.stop(),
            #[cfg(feature = "ffmpeg")]
            Source::File(mut f) => f.stop(),
        }
    }
}

use crate::device::{Backend, WorkerCtx};
use crate::output::Output;
use crate::stats::{StreamStats, redact_url};

/// Capacity of the packet channel between the async source and the decode
/// thread. When full, the source drops packets until the next keyframe.
const PACKET_QUEUE: usize = 96;

/// Frames older than this (receive → decoded) are dropped when gating is on.
const STALE_FRAME_NS: u64 = 1_000_000_000;

#[derive(Default)]
struct Counters {
    packets: AtomicU64,
    bytes: AtomicU64,
    decoded_packets: AtomicU64,
    skipped_packets: AtomicU64,
    frames: AtomicU64,
    decode_errors: AtomicU64,
    reconnects: AtomicU64,
    connected: AtomicBool,
}

#[derive(Default)]
struct Info {
    params: Option<StreamParams>,
    decoder: Option<String>,
    gate: &'static str,
    last_error: Option<String>,
    input_fps: f64,
    decode_fps: f64,
    latency_ms: f64,
}

/// State shared between the engine, the source and the decode thread.
pub(crate) struct StreamShared {
    pub(crate) cfg: StreamConfig,
    pub(crate) id: Arc<str>,
    counters: Counters,
    info: Mutex<Info>,
    force_until: AtomicU64,
    stop: AtomicBool,
}

impl StreamShared {
    pub(crate) fn stats(&self, source: Option<&Source>) -> StreamStats {
        let c = &self.counters;
        let info = self.info.lock();
        let (ingest_dropped, reconnects) = match source {
            Some(Source::Rtsp(s)) => {
                let st = s.stats();
                (st.dropped, st.reconnects)
            }
            _ => (0, c.reconnects.load(Ordering::Relaxed)),
        };
        StreamStats {
            id: self.cfg.id.clone(),
            url: redact_url(&self.cfg.url),
            connected: c.connected.load(Ordering::Relaxed),
            codec: info.params.as_ref().map(|p| p.codec.to_string()),
            width: info.params.as_ref().map_or(0, |p| p.width),
            height: info.params.as_ref().map_or(0, |p| p.height),
            decoder: info.decoder.clone(),
            gate: info.gate.to_string(),
            packets: c.packets.load(Ordering::Relaxed),
            bytes: c.bytes.load(Ordering::Relaxed),
            ingest_dropped,
            decoded_packets: c.decoded_packets.load(Ordering::Relaxed),
            skipped_packets: c.skipped_packets.load(Ordering::Relaxed),
            frames: c.frames.load(Ordering::Relaxed),
            decode_errors: c.decode_errors.load(Ordering::Relaxed),
            reconnects,
            input_fps: info.input_fps,
            decode_fps: info.decode_fps,
            avg_latency_ms: info.latency_ms,
            last_error: info.last_error.clone(),
        }
    }

    pub(crate) fn force_active(&self, d: Duration) {
        let until = now_ns().saturating_add(d.as_nanos() as u64);
        self.force_until.fetch_max(until, Ordering::Relaxed);
    }
}

/// A running stream.
pub(crate) struct StreamHandle {
    pub(crate) shared: Arc<StreamShared>,
    source: Option<Source>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl StreamHandle {
    pub(crate) fn stats(&self) -> StreamStats {
        self.shared.stats(self.source.as_ref())
    }

    pub(crate) fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(s) = self.source.take() {
            s.stop();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Packet metadata remembered until its frame comes out of the decoder.
#[derive(Clone, Copy)]
struct PacketInfo {
    pts: i64,
    seq: u64,
    keyframe: bool,
    recv_time_ns: u64,
    capture_time_ns: Option<u64>,
}

struct OutputLink {
    out: Arc<Output>,
    last_pts: Option<i64>,
    min_interval: Option<i64>,
}

/// Starts the source task and the decode thread of a stream.
pub(crate) fn spawn(
    cfg: StreamConfig,
    default_decoder: DecoderPref,
    outputs: Vec<Arc<Output>>,
    backend: Backend,
    rt: &tokio::runtime::Handle,
) -> Result<StreamHandle> {
    cfg.validate()?;
    let id: Arc<str> = cfg.id.as_str().into();
    let shared = Arc::new(StreamShared {
        cfg: cfg.clone(),
        id: id.clone(),
        counters: Counters::default(),
        info: Mutex::new(Info { gate: "warmup", ..Default::default() }),
        force_until: AtomicU64::new(0),
        stop: AtomicBool::new(false),
    });
    let (tx, rx) = tokio::sync::mpsc::channel(PACKET_QUEUE);
    let source = match cfg.source_kind()? {
        camtensor_core::SourceKind::Rtsp => {
            Source::Rtsp(RtspSource::spawn(rt, SourceConfig::from(&cfg), tx))
        }
        #[cfg(feature = "ffmpeg")]
        camtensor_core::SourceKind::File(path) => {
            Source::File(crate::filesrc::FileSource::spawn(id.clone(), path, cfg.playback, tx)?)
        }
        #[cfg(not(feature = "ffmpeg"))]
        camtensor_core::SourceKind::File(_) => {
            return Err(Error::unavailable("file sources need the `ffmpeg` feature"));
        }
    };
    let pref = cfg.decoder.unwrap_or(default_decoder);
    let thread = {
        let shared = shared.clone();
        std::thread::Builder::new()
            .name(format!("ct-stream-{}", cfg.id))
            .spawn(move || {
                let mut w = match Worker::new(shared.clone(), pref, outputs, backend) {
                    Ok(w) => w,
                    Err(e) => {
                        tracing::error!(stream = %shared.id, "cannot start worker: {e}");
                        shared.info.lock().last_error = Some(e.to_string());
                        return;
                    }
                };
                w.run(rx);
            })
            .map_err(Error::Io)?
    };
    Ok(StreamHandle { shared, source: Some(source), thread: Some(thread) })
}

struct Worker {
    shared: Arc<StreamShared>,
    pref: DecoderPref,
    backend: Backend,
    ctx: WorkerCtx,
    outputs: Vec<OutputLink>,
    decoder: Option<Box<dyn Decoder>>,
    params: Option<StreamParams>,
    gate: Option<Gate>,
    pending: VecDeque<PacketInfo>,
    feed: Vec<(EncodedPacket, bool)>,
    consecutive_errors: u32,
    // smoothing
    rate_window: Instant,
    win_packets: u64,
    win_frames: u64,
    latency_ema: f64,
}

impl Worker {
    fn new(
        shared: Arc<StreamShared>,
        pref: DecoderPref,
        outputs: Vec<Arc<Output>>,
        backend: Backend,
    ) -> Result<Self> {
        let ctx = backend.worker()?;
        let outputs = outputs
            .into_iter()
            .map(|out| {
                let min_interval =
                    out.cfg.max_fps.map(|f| (camtensor_core::time::PTS_CLOCK_HZ as f64 / f) as i64);
                OutputLink { out, last_pts: None, min_interval }
            })
            .collect();
        Ok(Self {
            shared,
            pref,
            backend,
            ctx,
            outputs,
            decoder: None,
            params: None,
            gate: None,
            pending: VecDeque::with_capacity(64),
            feed: Vec::with_capacity(64),
            consecutive_errors: 0,
            rate_window: Instant::now(),
            win_packets: 0,
            win_frames: 0,
            latency_ema: 0.0,
        })
    }

    fn run(&mut self, mut rx: tokio::sync::mpsc::Receiver<SourceEvent>) {
        while let Some(ev) = rx.blocking_recv() {
            if self.shared.stop.load(Ordering::Relaxed) {
                break;
            }
            match ev {
                SourceEvent::Connected(p) => {
                    self.shared.counters.connected.store(true, Ordering::Relaxed);
                    self.on_params(p);
                }
                SourceEvent::ParamsChanged(p) => self.on_params(p),
                SourceEvent::Packet(pkt) => self.on_packet(pkt),
                SourceEvent::Disconnected { reason } => {
                    let c = &self.shared.counters;
                    c.connected.store(false, Ordering::Relaxed);
                    c.reconnects.fetch_add(1, Ordering::Relaxed);
                    self.shared.info.lock().last_error = Some(reason);
                    // Restart gating: the next session must start at a keyframe.
                    self.gate = None;
                    // Deliver the pictures still held by the decoder (B-frame
                    // reordering), e.g. the last frames of a file.
                    let mut frames = Vec::new();
                    if let Some(d) = self.decoder.as_mut()
                        && let Err(e) = d.flush(&mut |f| frames.push(f))
                    {
                        tracing::debug!(stream = %self.shared.id, "flush: {e}");
                    }
                    for f in frames {
                        self.on_frame(f);
                    }
                }
            }
            self.update_rates();
        }
        if let Some(d) = self.decoder.as_mut() {
            let _ = d.flush(&mut |_| {});
        }
    }

    fn on_params(&mut self, p: StreamParams) {
        let reopen = match &self.params {
            Some(old) => {
                old.codec != p.codec
                    || old.width != p.width
                    || old.height != p.height
                    || old.extradata != p.extradata
            }
            None => true,
        };
        if reopen {
            self.decoder = None;
            self.gate = None;
        }
        self.shared.info.lock().params = Some(p.clone());
        self.params = Some(p);
    }

    fn open_decoder(&mut self) -> Result<()> {
        let params = self.params.as_ref().ok_or_else(|| Error::decode("no stream parameters"))?;
        let opts = DecoderOptions {
            pref: self.pref,
            device: self.backend.device(),
            // One thread per stream: frame threading adds (threads - 1) frames
            // of latency; streams already run in parallel.
            threads: 1,
            surfaces: 8,
        };
        let d = camtensor_decode::open(params, &opts)?;
        tracing::info!(stream = %self.shared.id, decoder = %d.kind(), codec = %params.codec,
            "{}x{}", params.width, params.height);
        self.shared.info.lock().decoder = Some(d.kind().to_string());
        self.decoder = Some(d);
        Ok(())
    }

    fn on_packet(&mut self, pkt: EncodedPacket) {
        let shared = self.shared.clone();
        let c = &shared.counters;
        c.packets.fetch_add(1, Ordering::Relaxed);
        c.bytes.fetch_add(pkt.data.len() as u64, Ordering::Relaxed);
        self.win_packets += 1;

        if self.params.is_none() {
            // Some sources only learn parameters from the first keyframe.
            self.params = Some(StreamParams {
                codec: pkt.codec,
                width: 0,
                height: 0,
                extradata: Default::default(),
                fps: None,
            });
        }
        if self.decoder.is_none()
            && let Err(e) = self.open_decoder()
        {
            self.record_error(&e);
            return;
        }
        let now = now_ns();
        let gate = self.gate.get_or_insert_with(|| Gate::new(self.shared.cfg.gating, pkt.codec));
        let force = self.shared.force_until.load(Ordering::Relaxed);
        if force > now {
            gate.force_active(force);
        }
        let mut feed = std::mem::take(&mut self.feed);
        feed.clear();
        gate.push(pkt, now, &mut feed);
        let gate_state = gate.state();
        let fed = feed.len() as u64;
        c.decoded_packets.fetch_add(fed, Ordering::Relaxed);
        if fed == 0 {
            c.skipped_packets.fetch_add(1, Ordering::Relaxed);
        }
        {
            let mut info = self.shared.info.lock();
            info.gate = match gate_state {
                GateState::Warmup => "warmup",
                GateState::Idle => "idle",
                GateState::Active => "active",
            };
        }

        for (p, emit) in feed.drain(..) {
            if emit {
                self.pending.push_back(PacketInfo {
                    pts: p.pts,
                    seq: p.seq,
                    keyframe: p.keyframe,
                    recv_time_ns: p.recv_time_ns,
                    capture_time_ns: p.capture_time_ns,
                });
                while self.pending.len() > 64 {
                    self.pending.pop_front();
                }
            }
            let mut frames: Vec<Frame> = Vec::new();
            let res = match self.decoder.as_mut() {
                Some(d) => d.decode(&p, emit, &mut |f| frames.push(f)),
                None => Ok(()),
            };
            match res {
                Ok(()) => self.consecutive_errors = 0,
                Err(e) => {
                    self.record_error(&e);
                    self.consecutive_errors += 1;
                    if self.consecutive_errors >= 10 {
                        // Reset the decoder and wait for the next keyframe.
                        self.decoder = None;
                        self.gate = None;
                        self.consecutive_errors = 0;
                        break;
                    }
                }
            }
            for f in frames {
                self.on_frame(f);
            }
        }
        self.feed = feed;
    }

    fn record_error(&self, e: &Error) {
        self.shared.counters.decode_errors.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(stream = %self.shared.id, "decode error: {e}");
        self.shared.info.lock().last_error = Some(e.to_string());
    }

    fn take_info(&mut self, pts: i64) -> PacketInfo {
        if let Some(pos) = self.pending.iter().position(|p| p.pts == pts)
            && let Some(i) = self.pending.remove(pos)
        {
            return i;
        }
        PacketInfo { pts, seq: 0, keyframe: false, recv_time_ns: now_ns(), capture_time_ns: None }
    }

    fn on_frame(&mut self, frame: Frame) {
        self.shared.counters.frames.fetch_add(1, Ordering::Relaxed);
        self.win_frames += 1;
        let info = self.take_info(frame.pts);
        let id = self.shared.id.clone();

        // With gating, a decoder that reorders (B-frames) may hold an idle
        // keyframe until the next one arrives; deliver fresh frames only.
        if self.shared.cfg.gating.mode != camtensor_core::GatingMode::Always
            && now_ns().saturating_sub(info.recv_time_ns) > STALE_FRAME_NS
        {
            self.shared.counters.skipped_packets.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Upload once for every CUDA output when the decoder produced host frames.
        let mut device_frame: Option<Frame> = None;

        for i in 0..self.outputs.len() {
            let (out, due) = {
                let link = &mut self.outputs[i];
                let due = match (link.min_interval, link.last_pts) {
                    // 10% tolerance so 25 fps sources are not decimated by a 25 fps cap.
                    (Some(min), Some(last)) => frame.pts - last >= min * 9 / 10 || frame.pts < last,
                    _ => true,
                };
                if due {
                    link.last_pts = Some(frame.pts);
                }
                (link.out.clone(), due)
            };
            if !due {
                out.note_drop(&id, 1);
                continue;
            }
            let src = if out.device.is_cuda() && !frame.is_device() {
                if device_frame.is_none() {
                    match self.ctx.frame_for(&frame, out.device) {
                        Ok(f) => device_frame = Some(f),
                        Err(e) => {
                            self.record_error(&e);
                            continue;
                        }
                    }
                }
                device_frame.clone().expect("uploaded")
            } else {
                frame.clone()
            };
            let spec = *out.spec();
            let plan = match self.shared.cfg.region {
                // Region of interest: the output only sees that part of the
                // frame; `transform` still maps to full-frame pixels.
                Some(r) => {
                    let (x, y, w, h) = r.to_pixels(src.desc.width, src.desc.height);
                    camtensor_preprocess::plan_roi(&camtensor_core::Roi::Box { x, y, w, h }, &spec)
                        .unwrap_or_else(|_| {
                            camtensor_preprocess::plan(src.desc.width, src.desc.height, &spec)
                        })
                }
                None => camtensor_preprocess::plan(src.desc.width, src.desc.height, &spec),
            };
            let keep = out.cfg.keep_frame;
            let ctx = &mut self.ctx;
            let lat = &mut self.latency_ema;
            let res = out.submit(&id, |slot| {
                let fence = ctx.write_image(&src, &plan.sampling, &spec, slot, out.device)?;
                let ready = now_ns();
                let ms = ready.saturating_sub(info.recv_time_ns) as f64 / 1e6;
                *lat = if *lat == 0.0 { ms } else { *lat * 0.95 + ms * 0.05 };
                Ok((
                    FrameMeta {
                        stream_id: id.clone(),
                        seq: info.seq,
                        pts: frame.pts,
                        keyframe: info.keyframe,
                        src_width: src.desc.width,
                        src_height: src.desc.height,
                        transform: plan.dst_from_src,
                        capture_time_ns: info.capture_time_ns,
                        recv_time_ns: info.recv_time_ns,
                        ready_time_ns: ready,
                        dropped_since_last: 0,
                    },
                    keep.then(|| src.clone()),
                    fence,
                ))
            });
            if let Err(e) = res
                && !matches!(e, Error::Closed)
            {
                self.record_error(&e);
            }
        }
    }

    fn update_rates(&mut self) {
        let el = self.rate_window.elapsed();
        if el >= Duration::from_secs(1) {
            let s = el.as_secs_f64();
            let mut info = self.shared.info.lock();
            let a = 0.5;
            info.input_fps = info.input_fps * (1.0 - a) + (self.win_packets as f64 / s) * a;
            info.decode_fps = info.decode_fps * (1.0 - a) + (self.win_frames as f64 / s) * a;
            info.latency_ms = self.latency_ema;
            self.win_packets = 0;
            self.win_frames = 0;
            self.rate_window = Instant::now();
        }
    }
}
