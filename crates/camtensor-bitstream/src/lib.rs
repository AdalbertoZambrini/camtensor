//! H.264/H.265/JPEG bitstream parsing and compressed-domain motion gating.
//!
//! Everything in this crate is pure, allocation-light and deterministic:
//!
//! * [`annexb`]: Annex B NAL iteration, AVCC (length-prefixed) to Annex B
//!   conversion and `avcC` / `hvcC` decoder configuration records.
//! * [`bits`]: emulation-prevention removal and an Exp-Golomb bit reader.
//! * [`h264`] / [`h265`]: NAL unit types, keyframe detection and SPS parsing
//!   (dimensions, bit depth, chroma format, colour description, frame rate).
//! * [`jpeg`]: JPEG frame header (SOF) parsing.
//! * [`gate`]: motion gating that decides which packets to decode, using only
//!   packet sizes and picture types.
//! * [`probe_params`]: one-call summary of a stream's parameter sets.

#![forbid(unsafe_code)]

use std::fmt;

use camtensor_core::{Codec, ColorSpec};

pub mod annexb;
pub mod bits;
pub mod gate;
pub mod h264;
pub mod h265;
pub mod jpeg;

pub use gate::{Gate, GateState, GateStats};

/// Errors produced by the parsers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The input ended before the structure was complete.
    Truncated,
    /// The input is malformed; the message names the offending field.
    Invalid(&'static str),
    /// The input is valid but uses a feature this crate does not handle.
    Unsupported(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => f.write_str("bitstream truncated"),
            Error::Invalid(what) => write!(f, "invalid bitstream: {what}"),
            Error::Unsupported(what) => write!(f, "unsupported bitstream feature: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Codec-independent summary of a video stream's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    /// Display (cropped) width in pixels.
    pub width: u32,
    /// Display (cropped) height in pixels.
    pub height: u32,
    /// Luma bit depth.
    pub bit_depth: u8,
    /// Chroma format: 0 = monochrome, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4,
    /// 255 = other (JPEG 4:1:1, 4:4:0, CMYK, ...).
    pub chroma_format: u8,
    /// Colour description (signalled or codec default).
    pub color: ColorSpec,
    /// Frame rate if signalled in the bitstream.
    pub fps: Option<f64>,
    /// Codec profile (`profile_idc`; 0 for JPEG).
    pub profile: u8,
    /// Codec level (`level_idc`; 0 for JPEG).
    pub level: u8,
}

/// Extracts a [`VideoInfo`] from parameter sets or a keyframe.
///
/// * H.264 / H.265: `data` is Annex B containing at least an SPS (for
///   example `StreamParams::extradata` or a keyframe with parameter sets).
/// * MJPEG: `data` is a JPEG image (or at least its header up to SOF).
pub fn probe_params(codec: Codec, data: &[u8]) -> Result<VideoInfo> {
    match codec {
        Codec::H264 => {
            let sps = annexb::nal_units(data)
                .find(|n| h264::nal_unit_type(n) == Some(h264::NAL_SPS))
                .ok_or(Error::Invalid("no H.264 SPS found"))?;
            Ok(h264::Sps::parse(sps)?.info())
        }
        Codec::H265 => {
            let sps = annexb::nal_units(data)
                .find(|n| h265::nal_unit_type(n) == Some(h265::NAL_SPS))
                .ok_or(Error::Invalid("no H.265 SPS found"))?;
            Ok(h265::Sps::parse(sps)?.info())
        }
        Codec::Mjpeg => Ok(jpeg::FrameHeader::parse(data)?.info()),
    }
}

/// Returns `true` if `data` (Annex B for H.26x) contains a random access
/// picture: IDR for H.264, IRAP (16..=23) for H.265, always for MJPEG.
pub fn is_keyframe(codec: Codec, data: &[u8]) -> bool {
    match codec {
        Codec::H264 => h264::is_keyframe(data),
        Codec::H265 => h265::is_keyframe(data),
        Codec::Mjpeg => true,
    }
}

#[cfg(test)]
pub(crate) mod testdata {
    //! Parameter sets produced by FFmpeg (libx264 / libx265 / mjpeg) from
    //! `testsrc2`, used as known-answer vectors.

    pub fn hex(s: &str) -> Vec<u8> {
        let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        s.chunks(2)
            .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
            .collect()
    }

    /// 1280x720 25 fps, High, 4:2:0 8-bit, full range, BT.709.
    pub const H264_720P_SPS: &str = "6764001facd9405005bb016e020202800000030080000019078c18cb";
    pub const H264_720P_PPS: &str = "68ebe3cb22c0";
    /// 1918x1078 30 fps, High 4:2:2 10-bit, limited range, SMPTE 170M.
    pub const H264_422_10_SPS: &str =
        "677a0028b6cd940780227a8bc05a83030320000003002000000781e30632c0";
    /// 1920x1080 25 fps, interlaced (frame_mbs_only_flag = 0).
    pub const H264_1080I_SPS: &str = "67640028acd94078044fde0220000003002000000643e2c5b2c0";
    /// 640x360 15 fps, Constrained Baseline.
    pub const H264_BASELINE_SPS: &str = "6742c016d900a02ff970110000030001000003001e0f162e48";

    /// 1280x720 25 fps, Main, 8-bit.
    pub const H265_720P_VPS: &str = "40010c01ffff01600000030090000003000003005d959809";
    pub const H265_720P_SPS: &str =
        "42010101600000030090000003000003005da00280802d165959a4932bc05a020000030002000003003210";
    pub const H265_720P_PPS: &str = "4401c172b46240";
    /// 1918x1078 30 fps, Main 10, full range, BT.470BG matrix.
    pub const H265_MAIN10_SPS: &str = "420101022000000300900000030000030078a003c08010e754d96566924caf016e0a0a0a08000003000800000300f040";

    /// 640x480 4:2:0 JPEG header (SOI .. SOF0, tables included).
    pub const JPEG_420_HDR: &str = "ffd8ffe000104a46494600010200000100010000fffe00104c61766336332e32382e31303000ffdb0043000808080908090b0b0b0b0b0b0d0c0d0d0d0d0d0d0d0d0d0d0d0e0e0e1111110e0e0e0d0d0e0e10101111121312111111111313141414181817171c1c1d222229ffc000110801e0028003012200021100031100";
    /// 322x242 4:2:2 JPEG header (SOI .. SOF0, quantisation table only).
    pub const JPEG_422_HDR: &str = "ffd8ffe000104a46494600010200000100010000fffe00104c61766336332e32382e31303000ffdb0043000808080908090b0b0b0b0b0b0d0c0d0d0d0d0d0d0d0d0d0d0d0e0e0e1111110e0e0e0d0d0e0e10101111121312111111111313141414181817171c1c1d222229ffc000110800f2014203012200021200031200";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::*;
    use camtensor_core::{ColorMatrix, ColorRange};

    #[test]
    fn probe_h264_extradata() {
        let mut ed = vec![0, 0, 0, 1];
        ed.extend(hex(H264_720P_SPS));
        ed.extend([0, 0, 0, 1]);
        ed.extend(hex(H264_720P_PPS));
        let i = probe_params(Codec::H264, &ed).unwrap();
        assert_eq!((i.width, i.height, i.bit_depth, i.chroma_format), (1280, 720, 8, 1));
        assert_eq!(i.color, ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt709 });
        assert_eq!(i.fps, Some(25.0));
        assert_eq!((i.profile, i.level), (100, 31));
    }

    #[test]
    fn probe_errors() {
        assert!(probe_params(Codec::H264, &[0, 0, 1, 0x68, 0xeb]).is_err());
        assert!(probe_params(Codec::H265, &[]).is_err());
        assert!(probe_params(Codec::Mjpeg, &[0xff, 0xd8]).is_err());
    }
}
