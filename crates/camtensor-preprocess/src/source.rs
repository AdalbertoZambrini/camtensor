//! Validated, bounds-checked view of a host frame.

use camtensor_core::{ColorMatrix, ColorRange, Error, FrameDesc, PixelFormat, Result};

/// One sample channel of a frame: a 2-D grid of `w x h` bytes where sample
/// `(x, y)` lives at `data[y * pitch + x * step]`.
///
/// Invariant (checked in [`Chan::new`]): every sample index is in bounds, i.e.
/// `(h - 1) * pitch + row_len() <= data.len()`, and `row_len()` fits in a
/// `u32` so that tap tables can store indices compactly.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Chan<'a> {
    data: &'a [u8],
    pub pitch: usize,
    pub step: usize,
    pub w: usize,
    pub h: usize,
}

impl<'a> Chan<'a> {
    fn new(
        buf: &'a [u8],
        offset: usize,
        pitch: usize,
        step: usize,
        w: usize,
        h: usize,
    ) -> Result<Self> {
        debug_assert!(w > 0 && h > 0 && step > 0);
        let row_len = (w - 1).checked_mul(step).and_then(|v| v.checked_add(1));
        let need = (h - 1)
            .checked_mul(pitch)
            .and_then(|v| v.checked_add(row_len?))
            .and_then(|v| v.checked_add(offset));
        match (row_len, need) {
            (Some(rl), Some(need)) if need <= buf.len() && rl <= u32::MAX as usize => {
                Ok(Self { data: &buf[offset..], pitch, step, w, h })
            }
            _ => Err(Error::Preprocess(format!(
                "frame buffer too small or plane out of bounds ({} bytes)",
                buf.len()
            ))),
        }
    }

    /// Number of bytes spanned by one row of samples.
    #[inline]
    pub fn row_len(&self) -> usize {
        (self.w - 1) * self.step + 1
    }

    /// Row `y` (clamped by the caller) as a slice of exactly `row_len()` bytes;
    /// sample `x` is at index `x * step`.
    #[inline]
    pub fn row(&self, y: usize) -> &'a [u8] {
        let s = y * self.pitch;
        &self.data[s..s + self.row_len()]
    }
}

/// Y'CbCr to RGB constants (see the crate docs for the exact formulas).
#[derive(Debug, Clone, Copy)]
pub(crate) struct YuvCoeffs {
    pub y_off: f32,
    pub y_scale: f32,
    pub c_scale: f32,
    pub rv: f32,
    pub gu: f32,
    pub gv: f32,
    pub bu: f32,
}

impl YuvCoeffs {
    pub fn new(range: ColorRange, matrix: ColorMatrix) -> Self {
        let (y_off, y_scale, c_scale) = match range {
            ColorRange::Limited => (16.0, 255.0 / 219.0, 255.0 / 224.0),
            ColorRange::Full => (0.0, 1.0, 1.0),
        };
        let (rv, gu, gv, bu) = match matrix {
            ColorMatrix::Bt601 => (1.402, 0.344136, 0.714136, 1.772),
            ColorMatrix::Bt709 => (1.5748, 0.187324, 0.468124, 1.8556),
        };
        Self { y_off, y_scale, c_scale, rv, gu, gv, bu }
    }
}

/// A validated source frame.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Src<'a> {
    /// Planar or semi-planar Y'CbCr; `u`/`v` share dimensions and step.
    Yuv { y: Chan<'a>, u: Chan<'a>, v: Chan<'a>, shift: (u32, u32), k: YuvCoeffs },
    /// Single luma channel.
    Gray(Chan<'a>),
    /// Packed RGB; channels in R, G, B order.
    Rgb([Chan<'a>; 3]),
}

impl<'a> Src<'a> {
    pub fn new(desc: &FrameDesc, data: &'a [u8]) -> Result<Self> {
        if desc.width == 0 || desc.height == 0 {
            return Err(Error::Preprocess("source frame has zero size".into()));
        }
        let w = desc.width as usize;
        let h = desc.height as usize;
        let (sx, sy) = desc.format.chroma_shift();
        let cw = (w + (1 << sx) - 1) >> sx;
        let ch = (h + (1 << sy) - 1) >> sy;
        let p = &desc.planes;
        let check_pitch = |i: usize, row_bytes: usize| -> Result<()> {
            if p[i].pitch < row_bytes {
                return Err(Error::Preprocess(format!(
                    "plane {i} pitch {} is smaller than its row ({row_bytes} bytes)",
                    p[i].pitch
                )));
            }
            Ok(())
        };
        Ok(match desc.format {
            PixelFormat::Nv12 => {
                check_pitch(0, w)?;
                check_pitch(1, cw * 2)?;
                Src::Yuv {
                    y: Chan::new(data, p[0].offset, p[0].pitch, 1, w, h)?,
                    u: Chan::new(data, p[1].offset, p[1].pitch, 2, cw, ch)?,
                    v: Chan::new(data, p[1].offset.saturating_add(1), p[1].pitch, 2, cw, ch)?,
                    shift: (sx, sy),
                    k: YuvCoeffs::new(desc.color.range, desc.color.matrix),
                }
            }
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                check_pitch(0, w)?;
                check_pitch(1, cw)?;
                check_pitch(2, cw)?;
                Src::Yuv {
                    y: Chan::new(data, p[0].offset, p[0].pitch, 1, w, h)?,
                    u: Chan::new(data, p[1].offset, p[1].pitch, 1, cw, ch)?,
                    v: Chan::new(data, p[2].offset, p[2].pitch, 1, cw, ch)?,
                    shift: (sx, sy),
                    k: YuvCoeffs::new(desc.color.range, desc.color.matrix),
                }
            }
            PixelFormat::Gray8 => {
                check_pitch(0, w)?;
                Src::Gray(Chan::new(data, p[0].offset, p[0].pitch, 1, w, h)?)
            }
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
                check_pitch(0, w.checked_mul(3).unwrap_or(usize::MAX))?;
                let off = p[0].offset;
                let c = |i: usize| Chan::new(data, off.saturating_add(i), p[0].pitch, 3, w, h);
                if desc.format == PixelFormat::Rgb24 {
                    Src::Rgb([c(0)?, c(1)?, c(2)?])
                } else {
                    Src::Rgb([c(2)?, c(1)?, c(0)?])
                }
            }
        })
    }

    /// Luma (or first channel) dimensions.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Src::Yuv { y, .. } | Src::Gray(y) => (y.w, y.h),
            Src::Rgb(c) => (c[0].w, c[0].h),
        }
    }
}
