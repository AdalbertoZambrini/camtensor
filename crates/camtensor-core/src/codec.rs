//! Encoded media: codecs, stream parameters and access-unit packets.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Video codecs camtensor understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// Motion JPEG (each packet is a complete baseline JPEG image).
    Mjpeg,
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Codec::H264 => "h264",
            Codec::H265 => "h265",
            Codec::Mjpeg => "mjpeg",
        })
    }
}

/// Static parameters of a video stream, known after `DESCRIBE` or after the
/// first keyframe.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamParams {
    /// Codec of the stream.
    pub codec: Codec,
    /// Coded width in pixels (0 when unknown yet).
    pub width: u32,
    /// Coded height in pixels (0 when unknown yet).
    pub height: u32,
    /// Parameter sets (VPS/SPS/PPS) in Annex B form, start codes included.
    /// Empty for MJPEG.
    pub extradata: Bytes,
    /// Nominal frame rate if signalled (frames per second).
    pub fps: Option<f64>,
}

/// One encoded access unit (one picture).
///
/// * H.264 / H.265: Annex B byte stream (`00 00 00 01` start codes). When the
///   picture is a keyframe, the parameter sets are prepended so that a decoder
///   can start from any keyframe.
/// * MJPEG: a complete JPEG image (SOI .. EOI).
#[derive(Clone)]
pub struct EncodedPacket {
    /// Stream identifier as configured by the user.
    pub stream_id: Arc<str>,
    /// Codec of this packet.
    pub codec: Codec,
    /// Payload, see type-level docs.
    pub data: Bytes,
    /// Presentation timestamp in 90 kHz ticks.
    pub pts: i64,
    /// `true` for IDR / IRAP pictures and every MJPEG frame.
    pub keyframe: bool,
    /// Monotonic per-stream sequence number, starting at 0, incremented for
    /// every packet produced by the source (gaps = packets lost upstream).
    pub seq: u64,
    /// Wall-clock time (ns since Unix epoch) when the packet was received.
    pub recv_time_ns: u64,
    /// Wall-clock capture time derived from RTCP sender reports, if known.
    pub capture_time_ns: Option<u64>,
    /// `true` if the source detected data loss right before this packet.
    pub loss: bool,
}

impl fmt::Debug for EncodedPacket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodedPacket")
            .field("stream_id", &self.stream_id)
            .field("codec", &self.codec)
            .field("len", &self.data.len())
            .field("pts", &self.pts)
            .field("keyframe", &self.keyframe)
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}
