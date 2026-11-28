//! Video-file source: demuxes a file with libavformat and feeds the same
//! event channel as an RTSP source.
//!
//! Unlike a camera, a file never drops packets on backpressure: the reader
//! waits for the pipeline, so offline evaluation sees every frame. With
//! `playback.realtime` packets are paced at the file's frame rate.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;

use camtensor_bitstream::annexb;
use camtensor_core::time::now_ns;
use camtensor_core::{Codec, EncodedPacket, Error, PlaybackConfig, Result, StreamParams};
use camtensor_ingest::SourceEvent;

/// Handle of a running file source.
pub(crate) struct FileSource {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl FileSource {
    pub(crate) fn spawn(
        stream_id: Arc<str>,
        path: PathBuf,
        playback: PlaybackConfig,
        tx: tokio::sync::mpsc::Sender<SourceEvent>,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let thread = std::thread::Builder::new()
            .name(format!("ct-file-{stream_id}"))
            .spawn(move || {
                if let Err(e) = run(&stream_id, &path, playback, &tx, &s) {
                    tracing::warn!(stream = %stream_id, "file source: {e}");
                    let _ = tx.blocking_send(SourceEvent::Disconnected { reason: e.to_string() });
                }
            })
            .map_err(Error::Io)?;
        Ok(Self { stop, thread: Some(thread) })
    }

    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for FileSource {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Converts container packets to Annex B access units.
struct ToAnnexB {
    codec: Codec,
    /// NAL length size for AVCC/HVCC input, `None` for Annex B input.
    length_size: Option<usize>,
    /// Parameter sets (Annex B) prepended to keyframes.
    param_sets: Vec<u8>,
}

impl ToAnnexB {
    fn new(codec: Codec, extradata: &[u8]) -> Result<Self> {
        let bad = |e: camtensor_bitstream::Error| Error::Bitstream(e.to_string());
        let (param_sets, length_size) = match codec {
            Codec::Mjpeg => (Vec::new(), None),
            _ if extradata.is_empty() || annexb::starts_with_start_code(extradata) => {
                (extradata.to_vec(), None)
            }
            Codec::H264 => {
                let (ps, n) = annexb::avcc_config_to_annexb(extradata).map_err(bad)?;
                (ps, Some(n))
            }
            Codec::H265 => {
                let (ps, n) = annexb::hvcc_config_to_annexb(extradata).map_err(bad)?;
                (ps, Some(n))
            }
        };
        Ok(Self { codec, length_size, param_sets })
    }

    fn convert(&self, data: &[u8], keyframe: bool) -> Result<Vec<u8>> {
        if self.codec == Codec::Mjpeg {
            return Ok(data.to_vec());
        }
        let mut out = Vec::with_capacity(data.len() + self.param_sets.len() + 16);
        if keyframe {
            out.extend_from_slice(&self.param_sets);
        }
        match self.length_size {
            Some(n) => annexb::avcc_to_annexb(data, n, &mut out)
                .map_err(|e| Error::Bitstream(e.to_string()))?,
            None => out.extend_from_slice(data),
        }
        Ok(out)
    }
}

fn run(
    stream_id: &Arc<str>,
    path: &std::path::Path,
    playback: PlaybackConfig,
    tx: &tokio::sync::mpsc::Sender<SourceEvent>,
    stop: &AtomicBool,
) -> Result<()> {
    use camtensor_decode::ffmpeg::demux::Demuxer;

    let mut dmx = Demuxer::open(path)?;
    let info = dmx.info().clone();
    let conv = ToAnnexB::new(info.codec, &info.extradata)?;
    let frame_s = 1.0 / info.fps.unwrap_or(25.0).clamp(1.0, 240.0);

    let mut params_sent = false;
    let mut seq: u64 = 0;
    let mut loop_offset = 0.0f64;
    let mut last_pts = 0.0f64;
    let mut t0 = Instant::now();
    let send = |ev: SourceEvent| tx.blocking_send(ev).map_err(|_| Error::Closed);

    while !stop.load(Ordering::Relaxed) {
        let Some(p) = dmx.read()? else {
            if !playback.looped {
                send(SourceEvent::Disconnected { reason: "end of file".into() })?;
                return Ok(());
            }
            dmx.rewind()?;
            loop_offset = last_pts + frame_s;
            continue;
        };
        let pts_s = loop_offset + p.pts_seconds.unwrap_or(seq as f64 * frame_s);
        last_pts = last_pts.max(pts_s);
        let data = conv.convert(&p.data, p.keyframe)?;
        if !params_sent {
            let probe = match info.codec {
                Codec::Mjpeg => camtensor_bitstream::probe_params(Codec::Mjpeg, &data),
                c => camtensor_bitstream::probe_params(c, &data)
                    .or_else(|_| camtensor_bitstream::probe_params(c, &conv.param_sets)),
            };
            let (width, height) = probe.as_ref().map(|v| (v.width, v.height)).unwrap_or((0, 0));
            send(SourceEvent::Connected(StreamParams {
                codec: info.codec,
                width,
                height,
                extradata: Bytes::from(conv.param_sets.clone()),
                fps: info.fps,
            }))?;
            params_sent = true;
            t0 = Instant::now();
        }
        if playback.realtime {
            let due = t0 + Duration::from_secs_f64(pts_s.max(0.0));
            let now = Instant::now();
            if due > now {
                // Sleep in slices so stop requests are honoured quickly.
                let mut left = due - now;
                while left > Duration::ZERO && !stop.load(Ordering::Relaxed) {
                    let step = left.min(Duration::from_millis(50));
                    std::thread::sleep(step);
                    left = left.saturating_sub(step);
                }
            }
        }
        let pkt = EncodedPacket {
            stream_id: stream_id.clone(),
            codec: info.codec,
            data: Bytes::from(data),
            pts: (pts_s * camtensor_core::time::PTS_CLOCK_HZ as f64).round() as i64,
            keyframe: p.keyframe,
            seq,
            recv_time_ns: now_ns(),
            capture_time_ns: None,
            loss: false,
        };
        seq += 1;
        // Blocking send: a file never drops frames, it waits for the pipeline.
        send(SourceEvent::Packet(pkt))?;
    }
    Ok(())
}
