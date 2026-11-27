//! Per-image metadata delivered with every tensor slot.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::geometry::Affine2;

/// Metadata for one image of a batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameMeta {
    /// Stream the image came from.
    pub stream_id: Arc<str>,
    /// Sequence number of the source packet within its stream.
    pub seq: u64,
    /// Presentation timestamp, 90 kHz ticks.
    pub pts: i64,
    /// `true` if decoded from a keyframe.
    pub keyframe: bool,
    /// Source (decoded) frame width.
    pub src_width: u32,
    /// Source (decoded) frame height.
    pub src_height: u32,
    /// Continuous `dst_from_src` transform used to build this image. Use
    /// [`FrameMeta::to_source`] to map detections back to source pixels.
    pub transform: Affine2,
    /// Camera capture time from RTCP (ns since Unix epoch), if known.
    pub capture_time_ns: Option<u64>,
    /// When the packet was received (ns since Unix epoch).
    pub recv_time_ns: u64,
    /// When the image was ready in the tensor (ns since Unix epoch).
    pub ready_time_ns: u64,
    /// Frames of this stream dropped for this output since the previous
    /// delivered image (backpressure, `max_fps`, gating excluded).
    pub dropped_since_last: u64,
}

impl FrameMeta {
    /// Maps a point from tensor pixels back to source-frame pixels.
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        match self.transform.invert() {
            Some(inv) => inv.apply(x, y),
            None => (x, y),
        }
    }

    /// Maps an axis-aligned box `(x1, y1, x2, y2)` from tensor pixels back to
    /// source pixels.
    pub fn box_to_source(&self, b: [f32; 4]) -> [f32; 4] {
        let (x1, y1) = self.to_source(b[0], b[1]);
        let (x2, y2) = self.to_source(b[2], b[3]);
        [x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)]
    }

    /// End-to-end latency in milliseconds (capture → ready when known,
    /// receive → ready otherwise).
    pub fn latency_ms(&self) -> f64 {
        let start = self.capture_time_ns.unwrap_or(self.recv_time_ns);
        self.ready_time_ns.saturating_sub(start) as f64 / 1e6
    }
}
