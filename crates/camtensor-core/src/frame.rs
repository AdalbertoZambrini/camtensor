//! Decoded-frame description (pixel formats, plane layout, colour space).

use serde::{Deserialize, Serialize};

/// Pixel formats produced by camtensor decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PixelFormat {
    /// 8-bit Y plane + interleaved UV plane at half resolution (NVDEC native).
    Nv12,
    /// 8-bit planar Y, U, V; chroma at half width and half height.
    Yuv420p,
    /// 8-bit planar Y, U, V; chroma at half width, full height.
    Yuv422p,
    /// 8-bit planar Y, U, V; chroma at full resolution.
    Yuv444p,
    /// 8-bit luma only.
    Gray8,
    /// 8-bit packed R, G, B.
    Rgb24,
    /// 8-bit packed B, G, R.
    Bgr24,
}

impl PixelFormat {
    /// Number of planes.
    pub fn plane_count(self) -> usize {
        match self {
            PixelFormat::Nv12 => 2,
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => 3,
            PixelFormat::Gray8 | PixelFormat::Rgb24 | PixelFormat::Bgr24 => 1,
        }
    }

    /// Chroma subsampling as `(shift_x, shift_y)` (log2 of the ratio).
    pub fn chroma_shift(self) -> (u32, u32) {
        match self {
            PixelFormat::Nv12 | PixelFormat::Yuv420p => (1, 1),
            PixelFormat::Yuv422p => (1, 0),
            _ => (0, 0),
        }
    }

    /// `true` for formats that carry Y'CbCr samples.
    pub fn is_yuv(self) -> bool {
        matches!(
            self,
            PixelFormat::Nv12 | PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p
        )
    }
}

/// Quantisation range of Y'CbCr samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColorRange {
    /// "TV" range: Y in 16..=235, chroma in 16..=240.
    #[default]
    Limited,
    /// "PC"/JPEG range: 0..=255.
    Full,
}

/// Y'CbCr to RGB matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColorMatrix {
    /// ITU-R BT.601 (SD, JPEG).
    Bt601,
    /// ITU-R BT.709 (HD). Default for H.264/H.265 when unsignalled.
    #[default]
    Bt709,
}

/// Colour description of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ColorSpec {
    /// Quantisation range.
    pub range: ColorRange,
    /// Conversion matrix.
    pub matrix: ColorMatrix,
}

/// One plane of a frame inside its buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlaneDesc {
    /// Byte offset of the plane from the start of the buffer.
    pub offset: usize,
    /// Row pitch in bytes.
    pub pitch: usize,
}

/// Full description of a decoded frame stored in a single buffer
/// (host or device memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDesc {
    /// Visible width in pixels.
    pub width: u32,
    /// Visible height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Colour description.
    pub color: ColorSpec,
    /// Plane layout; only the first `format.plane_count()` entries are valid.
    pub planes: [PlaneDesc; 3],
}

impl FrameDesc {
    /// Builds a tightly packed description (pitch = row bytes, planes
    /// contiguous) for the given size and format.
    pub fn packed(width: u32, height: u32, format: PixelFormat, color: ColorSpec) -> Self {
        let w = width as usize;
        let h = height as usize;
        let (sx, sy) = format.chroma_shift();
        let cw = (w + (1 << sx) - 1) >> sx;
        let ch = (h + (1 << sy) - 1) >> sy;
        let mut planes = [PlaneDesc::default(); 3];
        match format {
            PixelFormat::Nv12 => {
                planes[0] = PlaneDesc { offset: 0, pitch: w };
                planes[1] = PlaneDesc { offset: w * h, pitch: cw * 2 };
            }
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                planes[0] = PlaneDesc { offset: 0, pitch: w };
                planes[1] = PlaneDesc { offset: w * h, pitch: cw };
                planes[2] = PlaneDesc { offset: w * h + cw * ch, pitch: cw };
            }
            PixelFormat::Gray8 => planes[0] = PlaneDesc { offset: 0, pitch: w },
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
                planes[0] = PlaneDesc { offset: 0, pitch: w * 3 }
            }
        }
        Self { width, height, format, color, planes }
    }

    /// Minimum buffer size in bytes needed to hold this frame.
    pub fn byte_len(&self) -> usize {
        let h = self.height as usize;
        let (_, sy) = self.format.chroma_shift();
        let ch = (h + (1 << sy) - 1) >> sy;
        match self.format {
            PixelFormat::Nv12 => self.planes[1].offset + self.planes[1].pitch * ch,
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                self.planes[2].offset + self.planes[2].pitch * ch
            }
            PixelFormat::Gray8 | PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
                self.planes[0].offset + self.planes[0].pitch * h
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_sizes() {
        let c = ColorSpec::default();
        assert_eq!(FrameDesc::packed(4, 4, PixelFormat::Nv12, c).byte_len(), 24);
        assert_eq!(FrameDesc::packed(4, 4, PixelFormat::Yuv420p, c).byte_len(), 24);
        assert_eq!(FrameDesc::packed(5, 3, PixelFormat::Yuv420p, c).byte_len(), 15 + 2 * 6);
        assert_eq!(FrameDesc::packed(4, 4, PixelFormat::Yuv422p, c).byte_len(), 32);
        assert_eq!(FrameDesc::packed(4, 4, PixelFormat::Yuv444p, c).byte_len(), 48);
        assert_eq!(FrameDesc::packed(4, 4, PixelFormat::Rgb24, c).byte_len(), 48);
    }
}

/// Device (GPU) memory owned by some backend. Implemented by
/// `camtensor-cuda`; kept as a trait here so that `camtensor-core` has no
/// GPU dependency.
pub trait DeviceMemory: Send + Sync + std::fmt::Debug {
    /// Raw device pointer (e.g. `CUdeviceptr`).
    fn device_ptr(&self) -> u64;
    /// Size in bytes.
    fn byte_len(&self) -> usize;
    /// Device ordinal.
    fn device_ordinal(&self) -> u32;
    /// Downcasting support.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Storage of a decoded frame.
#[derive(Debug, Clone)]
pub enum FrameData {
    /// Host memory.
    Host(std::sync::Arc<Vec<u8>>),
    /// Device memory.
    Device(std::sync::Arc<dyn DeviceMemory>),
}

/// A decoded picture: description + storage + presentation timestamp.
///
/// Frames are cheap to clone (reference counted) and immutable once
/// produced.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Layout of the picture inside `data`.
    pub desc: FrameDesc,
    /// Pixel storage.
    pub data: FrameData,
    /// Presentation timestamp (90 kHz) of the packet that produced it.
    pub pts: i64,
}

impl Frame {
    /// Host bytes if the frame lives in host memory.
    pub fn host_bytes(&self) -> Option<&[u8]> {
        match &self.data {
            FrameData::Host(v) => Some(v.as_slice()),
            FrameData::Device(_) => None,
        }
    }

    /// `true` if the frame lives in device memory.
    pub fn is_device(&self) -> bool {
        matches!(self.data, FrameData::Device(_))
    }
}
