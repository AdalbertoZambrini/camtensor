//! Connects to an RTSP stream and prints what camtensor sees.
//!
//! ```text
//! cargo run -p camtensor-ingest --example probe -- rtsp://127.0.0.1:8554/cam1 [seconds] [--udp]
//! ```

use std::time::{Duration, Instant};

use camtensor_bitstream::{h264, probe_params};
use camtensor_core::{Codec, StreamParams, Transport};
use camtensor_ingest::{RtspSource, SourceConfig, SourceEvent};
use tokio::sync::mpsc;

#[derive(Default)]
struct Sizes {
    n: u64,
    total: u64,
    min: u64,
    max: u64,
}

impl Sizes {
    fn add(&mut self, len: usize) {
        let len = len as u64;
        self.min = if self.n == 0 { len } else { self.min.min(len) };
        self.max = self.max.max(len);
        self.n += 1;
        self.total += len;
    }

    fn line(&self) -> String {
        if self.n == 0 {
            return "-".into();
        }
        format!("n={} min={} avg={} max={} bytes", self.n, self.min, self.total / self.n, self.max)
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(url) = args.iter().find(|a| !a.starts_with("--") && a.contains("://")).cloned() else {
        eprintln!("usage: probe <rtsp-url> [seconds] [--udp]");
        std::process::exit(2);
    };
    let secs: u64 = args.iter().filter_map(|a| a.parse().ok()).next().unwrap_or(10);
    let mut cfg = SourceConfig::new("probe", url);
    if args.iter().any(|a| a == "--udp") {
        cfg.transport = Transport::Udp;
    }
    let (tx, mut rx) = mpsc::channel(512);
    let handle = RtspSource::spawn(&tokio::runtime::Handle::current(), cfg, tx);

    let mut params: Option<StreamParams> = None;
    let (mut key, mut delta) = (Sizes::default(), Sizes::default());
    let mut key_seqs = Vec::new();
    let mut key_pts = Vec::new();
    let (mut first_pts, mut last_pts) = (None, 0i64);
    let mut first_key: Option<bytes::Bytes> = None;
    let mut slice_types = std::collections::BTreeMap::new();
    let mut capture_lat = Vec::new();
    let mut losses = 0u64;
    let start = Instant::now();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);

    while let Ok(Some(ev)) = tokio::time::timeout_at(deadline, rx.recv()).await {
        match ev {
            SourceEvent::Connected(p) => {
                println!("connected after {:.2} s", start.elapsed().as_secs_f64());
                params = Some(p);
            }
            SourceEvent::ParamsChanged(p) => {
                println!("parameters changed: {}x{}", p.width, p.height);
                params = Some(p);
            }
            SourceEvent::Disconnected { reason } => println!("disconnected: {reason}"),
            SourceEvent::Packet(p) => {
                first_pts.get_or_insert(p.pts);
                last_pts = p.pts;
                losses += u64::from(p.loss);
                if let Some(c) = p.capture_time_ns {
                    capture_lat.push(p.recv_time_ns as f64 / 1e6 - c as f64 / 1e6);
                }
                if p.codec == Codec::H264
                    && let Some(t) = h264::first_slice_type(&p.data)
                {
                    *slice_types.entry(format!("{t:?}")).or_insert(0u64) += 1;
                }
                if p.keyframe {
                    key.add(p.data.len());
                    key_seqs.push(p.seq);
                    key_pts.push(p.pts);
                    if first_key.is_none() {
                        first_key = Some(p.data.clone());
                    }
                } else {
                    delta.add(p.data.len());
                }
            }
        }
    }
    let stats = handle.stats();
    handle.join().await;

    let Some(p) = params else {
        println!("no stream parameters received ({:?})", stats.last_error);
        std::process::exit(1);
    };
    println!("codec:        {}", p.codec);
    println!("resolution:   {}x{}", p.width, p.height);
    let probe_src =
        if p.extradata.is_empty() { first_key.as_deref() } else { Some(&p.extradata[..]) };
    match probe_src.map(|d| probe_params(p.codec, d)) {
        Some(Ok(i)) => println!(
            "bitstream:    {}x{} {}-bit chroma_format={} profile={} level={} colour={:?}/{:?}",
            i.width,
            i.height,
            i.bit_depth,
            i.chroma_format,
            i.profile,
            i.level,
            i.color.range,
            i.color.matrix
        ),
        Some(Err(e)) => println!("bitstream:    parse error: {e}"),
        None => println!("bitstream:    no parameter sets / keyframe seen"),
    }
    let frames = key.n + delta.n;
    let dur_s = (last_pts - first_pts.unwrap_or(last_pts)) as f64 / 90_000.0;
    let measured = (frames > 1 && dur_s > 0.0).then(|| (frames - 1) as f64 / dur_s);
    println!(
        "fps:          signalled {} / measured {}",
        p.fps.map_or("-".into(), |f| format!("{f:.3}")),
        measured.map_or("-".into(), |f| format!("{f:.3}"))
    );
    if dur_s > 0.0 {
        let kbps = (key.total + delta.total) as f64 * 8.0 / dur_s / 1000.0;
        println!("bitrate:      {kbps:.0} kbit/s over {dur_s:.1} s");
    }
    if key_seqs.len() >= 2 {
        let n = (key_seqs.len() - 1) as f64;
        let frames_per_gop = (key_seqs[key_seqs.len() - 1] - key_seqs[0]) as f64 / n;
        let secs_per_gop = (key_pts[key_pts.len() - 1] - key_pts[0]) as f64 / n / 90_000.0;
        println!("keyframes:    every {frames_per_gop:.1} frames ({secs_per_gop:.2} s)");
    } else {
        println!("keyframes:    {} seen (interval unknown)", key_seqs.len());
    }
    println!("key sizes:    {}", key.line());
    println!("delta sizes:  {}", delta.line());
    if !slice_types.is_empty() {
        println!("slice types:  {slice_types:?}");
    }
    if !capture_lat.is_empty() {
        capture_lat.sort_by(f64::total_cmp);
        println!(
            "rtcp latency: median {:.1} ms (receive - capture, {} packets)",
            capture_lat[capture_lat.len() / 2],
            capture_lat.len()
        );
    } else {
        println!("rtcp latency: no sender report received");
    }
    println!(
        "source:       packets={} bytes={} dropped={} loss_flags={} reconnects={}",
        stats.packets, stats.bytes, stats.dropped, losses, stats.reconnects
    );
}
