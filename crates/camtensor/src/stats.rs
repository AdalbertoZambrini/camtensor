//! Runtime statistics exposed by the engine (and by the service as JSON and
//! Prometheus metrics).

use serde::Serialize;
use std::collections::BTreeMap;

use crate::output::OutputStats;

/// Statistics of one stream.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StreamStats {
    /// Stream id.
    pub id: String,
    /// URL with credentials removed.
    pub url: String,
    /// `true` while the RTSP session is playing.
    pub connected: bool,
    /// Codec once known.
    pub codec: Option<String>,
    /// Decoded width.
    pub width: u32,
    /// Decoded height.
    pub height: u32,
    /// Decoder in use (`nvdec`, `ffmpeg`, `jpeg`).
    pub decoder: Option<String>,
    /// Gate state (`warmup`, `idle`, `active`).
    pub gate: String,
    /// Packets received.
    pub packets: u64,
    /// Bytes received.
    pub bytes: u64,
    /// Packets dropped by ingest backpressure.
    pub ingest_dropped: u64,
    /// Packets fed to the decoder.
    pub decoded_packets: u64,
    /// Packets skipped by gating (never decoded).
    pub skipped_packets: u64,
    /// Frames emitted by the decoder.
    pub frames: u64,
    /// Decoder errors.
    pub decode_errors: u64,
    /// Reconnections.
    pub reconnects: u64,
    /// Received packets per second (smoothed).
    pub input_fps: f64,
    /// Decoded frames per second (smoothed).
    pub decode_fps: f64,
    /// Average time from packet receipt to image ready, in milliseconds.
    pub avg_latency_ms: f64,
    /// Last error, if any.
    pub last_error: Option<String>,
}

/// Statistics of the whole engine.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EngineStats {
    /// Device in use.
    pub device: String,
    /// Per-stream statistics.
    pub streams: Vec<StreamStats>,
    /// Per-output statistics.
    pub outputs: BTreeMap<String, OutputStats>,
}

impl EngineStats {
    /// Renders the statistics in Prometheus text exposition format.
    pub fn to_prometheus(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let mut metric = |name: &str, help: &str, kind: &str, rows: Vec<(String, f64)>| {
            let _ = writeln!(s, "# HELP camtensor_{name} {help}");
            let _ = writeln!(s, "# TYPE camtensor_{name} {kind}");
            for (labels, v) in rows {
                let _ = writeln!(s, "camtensor_{name}{{{labels}}} {v}");
            }
        };
        let st = |f: &dyn Fn(&StreamStats) -> f64| -> Vec<(String, f64)> {
            self.streams.iter().map(|x| (format!("stream=\"{}\"", esc(&x.id)), f(x))).collect()
        };
        metric("stream_connected", "1 when the RTSP session is playing", "gauge", st(&|x| x.connected as u8 as f64));
        metric("stream_packets_total", "Packets received", "counter", st(&|x| x.packets as f64));
        metric("stream_bytes_total", "Bytes received", "counter", st(&|x| x.bytes as f64));
        metric("stream_decoded_packets_total", "Packets decoded", "counter", st(&|x| x.decoded_packets as f64));
        metric("stream_skipped_packets_total", "Packets skipped by gating", "counter", st(&|x| x.skipped_packets as f64));
        metric("stream_frames_total", "Frames decoded", "counter", st(&|x| x.frames as f64));
        metric("stream_decode_errors_total", "Decoder errors", "counter", st(&|x| x.decode_errors as f64));
        metric("stream_ingest_dropped_total", "Packets dropped by ingest backpressure", "counter", st(&|x| x.ingest_dropped as f64));
        metric("stream_reconnects_total", "Reconnections", "counter", st(&|x| x.reconnects as f64));
        metric("stream_input_fps", "Received packets per second", "gauge", st(&|x| x.input_fps));
        metric("stream_decode_fps", "Decoded frames per second", "gauge", st(&|x| x.decode_fps));
        metric("stream_latency_ms", "Average receive-to-ready latency", "gauge", st(&|x| x.avg_latency_ms));
        let out = |f: &dyn Fn(&OutputStats) -> f64| -> Vec<(String, f64)> {
            self.outputs.iter().map(|(k, v)| (format!("output=\"{}\"", esc(k)), f(v))).collect()
        };
        metric("output_batches_total", "Batches delivered", "counter", out(&|x| x.batches as f64));
        metric("output_images_total", "Images delivered", "counter", out(&|x| x.images as f64));
        metric("output_dropped_images_total", "Images dropped by backpressure", "counter", out(&|x| x.dropped_images as f64));
        metric("output_queued_batches", "Batches waiting for the consumer", "gauge", out(&|x| x.queued as f64));
        s
    }
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Removes `user:pass@` from a URL for display.
pub fn redact_url(url: &str) -> String {
    match (url.find("://"), url.rfind('@')) {
        (Some(scheme), Some(at)) if at > scheme => format!("{}{}", &url[..scheme + 3], &url[at + 1..]),
        _ => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_credentials() {
        assert_eq!(redact_url("rtsp://u:p@host:554/x"), "rtsp://host:554/x");
        assert_eq!(redact_url("rtsp://host/x"), "rtsp://host/x");
    }

    #[test]
    fn prometheus_output() {
        let s = EngineStats {
            device: "cpu".into(),
            streams: vec![StreamStats { id: "cam\"1".into(), packets: 5, ..Default::default() }],
            outputs: [("det".to_string(), OutputStats { images: 3, ..Default::default() })].into(),
        };
        let text = s.to_prometheus();
        assert!(text.contains("camtensor_stream_packets_total{stream=\"cam\\\"1\"} 5"));
        assert!(text.contains("camtensor_output_images_total{output=\"det\"} 3"));
    }
}
