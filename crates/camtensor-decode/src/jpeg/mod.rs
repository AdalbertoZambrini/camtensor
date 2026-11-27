//! Pure-Rust MJPEG decoder based on zune-jpeg.
//!
//! Every packet is a complete JPEG image; it is decoded to packed
//! [`PixelFormat::Rgb24`] (full range, JFIF BT.601 conversion done by
//! zune-jpeg) or [`PixelFormat::Gray8`] for greyscale JPEGs. zune-jpeg can
//! also output YCbCr, but only chroma-upsampled and interleaved, which is
//! neither cheaper nor a format camtensor consumes, so RGB is used.

// Shared with the ffmpeg backend without touching `lib.rs`; either feature
// can be enabled alone.
#[allow(clippy::duplicate_mod)]
#[path = "../ffmpeg/pool.rs"]
mod pool;

use camtensor_core::{
    ColorMatrix, ColorRange, ColorSpec, Device, EncodedPacket, Frame, FrameData, FrameDesc,
    PixelFormat, Result,
};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;

use crate::{Decoder, DecoderKind};
use pool::BufPool;

/// Largest accepted image side (zune-jpeg's default limit is lower).
const MAX_DIM: usize = 16384;

/// zune-jpeg based MJPEG decoder.
#[derive(Debug)]
pub struct JpegDecoder {
    pool: BufPool,
    options: DecoderOptions,
}

impl Default for JpegDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl JpegDecoder {
    /// Creates a decoder.
    pub fn new() -> Self {
        Self {
            pool: BufPool::new(4),
            // `new_fast` enables the SIMD paths (the default options are the
            // portable "safe" ones).
            options: DecoderOptions::new_fast().set_max_width(MAX_DIM).set_max_height(MAX_DIM),
        }
    }

    /// Decodes one JPEG image; `Ok(None)` if it is corrupt.
    fn decode_image(&mut self, data: &[u8], pts: i64) -> Option<Frame> {
        let mut dec = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(data), self.options);
        if let Err(e) = dec.decode_headers() {
            tracing::debug!(error = %e, "dropping corrupt JPEG (headers)");
            return None;
        }
        let gray = dec.input_colorspace() == Some(ColorSpace::Luma);
        let (out_cs, format) = if gray {
            (ColorSpace::Luma, PixelFormat::Gray8)
        } else {
            (ColorSpace::RGB, PixelFormat::Rgb24)
        };
        dec.set_options(self.options.jpeg_set_out_colorspace(out_cs));
        let (w, h) = dec.dimensions()?;
        let color = ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt601 };
        let desc = FrameDesc::packed(w as u32, h as u32, format, color);
        let mut buf = self.pool.take(desc.byte_len());
        if dec.output_buffer_size() != Some(buf.len()) {
            tracing::debug!("unexpected JPEG output size");
            return None;
        }
        if let Err(e) = dec.decode_into(&mut buf) {
            tracing::debug!(error = %e, "dropping corrupt JPEG");
            return None;
        }
        Some(Frame { desc, data: FrameData::Host(self.pool.share(buf)), pts })
    }
}

impl Decoder for JpegDecoder {
    fn decode(
        &mut self,
        pkt: &EncodedPacket,
        emit: bool,
        sink: &mut dyn FnMut(Frame),
    ) -> Result<()> {
        // Intra-only: a picture that is not emitted needs no decoding.
        if !emit || pkt.data.is_empty() {
            return Ok(());
        }
        if let Some(frame) = self.decode_image(&pkt.data, pkt.pts) {
            sink(frame);
        }
        Ok(())
    }

    fn flush(&mut self, _: &mut dyn FnMut(Frame)) -> Result<()> {
        Ok(())
    }

    fn kind(&self) -> DecoderKind {
        DecoderKind::Jpeg
    }

    fn device(&self) -> Device {
        Device::Cpu
    }
}
