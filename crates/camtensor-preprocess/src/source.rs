//! Validated, bounds-checked view of a host frame.

use camtensor_core::{ColorMatrix, ColorRange, Error, FrameDesc, PixelFormat, Result};

/// One plane of a frame as a 2-D grid of `w x h` samples, each made of
/// `comps` interleaved bytes: component `k` of sample `(x, y)` lives at
/// `data[y * pitch + x * step + k]`.
///
/// Invariant (checked in [`Chan::new`]): every component of every sample is
/// in bounds, i.e. `(h - 1) * pitch + row_len() <= data.len()`, and that
/// extent fits in a `u32` so tap tables can store byte offsets compactly
/// (planes larger than 4 GiB are rejected).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Chan<'a> {
    data: &'a [u8],
    pub pitch: usize,
    pub step: usize,
    pub comps: usize,
    pub w: usize,
    pub h: usize,
}

impl<'a> Chan<'a> {
    fn new(
        buf: &'a [u8],
        offset: usize,
        pitch: usize,
        (step, comps): (usize, usize),
        w: usize,
        h: usize,
    ) -> Result<Self> {
        debug_assert!(w > 0 && h > 0 && comps > 0 && step >= comps);
        let row_len = (w - 1).checked_mul(step).and_then(|v| v.checked_add(comps));
        let extent = (h - 1).checked_mul(pitch).and_then(|v| v.checked_add(row_len?));
        let need = extent.and_then(|v| v.checked_add(offset));
        match (extent, need) {
            (Some(ext), Some(need)) if need <= buf.len() && ext <= u32::MAX as usize => {
                Ok(Self { data: &buf[offset..], pitch, step, comps, w, h })
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
        (self.w - 1) * self.step + self.comps
    }

    /// Row `y < h` as a slice of exactly `row_len()` bytes.
    #[inline]
    pub fn row(&self, y: usize) -> &'a [u8] {
        let s = y * self.pitch;
        &self.data[s..s + self.row_len()]
    }

    /// Computes per-pixel taps for source index coordinates `(sx[i], sy[i])`
    /// (general path). Vectorisable: no branches, 32-bit arithmetic.
    pub fn taps(&self, bilinear: bool, sx: &[f32], sy: &[f32], t: &mut Taps2) {
        let n = sx.len();
        let (wm, hm) = (self.w as i32 - 1, self.h as i32 - 1);
        // Fit in u32 by the struct invariant (when h == 1 the row index is 0).
        let (step, pitch) = (self.step as u32, self.pitch as u32);
        let (sy, o, dx, dy) = (&sy[..n], &mut t.o[..n], &mut t.dx[..n], &mut t.dy[..n]);
        let (fx, fy) = (&mut t.fx[..n], &mut t.fy[..n]);
        if bilinear {
            for i in 0..n {
                let x0 = floor_i32(sx[i]);
                let y0 = floor_i32(sy[i]);
                fx[i] = sx[i] - x0 as f32;
                fy[i] = sy[i] - y0 as f32;
                // Clamp to [-1, max] first so `+ 1` cannot overflow; taps
                // are then clamped to [0, max] (`xb >= xa` always).
                let (x0c, y0c) = (x0.max(-1).min(wm), y0.max(-1).min(hm));
                let (xa, xb) = (x0c.max(0) as u32, (x0c + 1).min(wm) as u32);
                let (ya, yb) = (y0c.max(0) as u32, (y0c + 1).min(hm) as u32);
                o[i] = ya * pitch + xa * step;
                dx[i] = (xb - xa) * step;
                dy[i] = (yb - ya) * pitch;
            }
        } else {
            for i in 0..n {
                let xi = floor_i32(sx[i] + 0.5).max(0).min(wm) as u32;
                let yi = floor_i32(sy[i] + 0.5).max(0).min(hm) as u32;
                o[i] = yi * pitch + xi * step;
                dx[i] = 0;
                dy[i] = 0;
                fx[i] = 0.0;
                fy[i] = 0.0;
            }
        }
        // Largest byte offset any tap reads (before adding the component).
        t.n = n;
        t.max = o.iter().zip(&*dx).zip(&*dy).map(|((&o, &x), &y)| o + x + y).max().unwrap_or(0)
            as usize;
    }

    /// `out[k][i]` = component `k` sampled with taps `t[i]` (as computed by
    /// [`Chan::taps`] for this channel's geometry).
    #[inline(always)]
    pub fn gather<const N: usize>(&self, bilinear: bool, t: &Taps2, out: [&mut [f32]; N]) {
        let n = t.n;
        // Establishes the bound for the unchecked loads below: every offset
        // read is `<= t.max + N - 1 < data.len()`.
        assert!(N <= self.comps && t.max + N <= self.data.len());
        for o in &out {
            assert!(o.len() >= n);
        }
        let (o, dx, dy, fx, fy) = (&t.o[..n], &t.dx[..n], &t.dy[..n], &t.fx[..n], &t.fy[..n]);
        let d = self.data;
        let mut out = out;
        for i in 0..n {
            let a = o[i] as usize;
            if bilinear {
                let (b, c) = (a + dx[i] as usize, a + dy[i] as usize);
                let e = c + dx[i] as usize;
                for (k, out) in out.iter_mut().enumerate() {
                    // SAFETY: `a <= b, c <= e = o + dx + dy <= t.max` (max
                    // taken over the same arrays) and `k < N`, so every index
                    // is below `t.max + N <= d.len()` (asserted above).
                    let (p00, p01, p10, p11) = unsafe {
                        (
                            *d.get_unchecked(a + k),
                            *d.get_unchecked(b + k),
                            *d.get_unchecked(c + k),
                            *d.get_unchecked(e + k),
                        )
                    };
                    let top = lerp(p00 as f32, p01 as f32, fx[i]);
                    let bot = lerp(p10 as f32, p11 as f32, fx[i]);
                    out[i] = lerp(top, bot, fy[i]);
                }
            } else {
                for (k, out) in out.iter_mut().enumerate() {
                    // SAFETY: `a <= t.max` and `k < N` (see above).
                    out[i] = unsafe { *d.get_unchecked(a + k) } as f32;
                }
            }
        }
    }
}

/// Per-pixel 2-D taps of the general path: sample `(x0, y0)` at byte offset
/// `o`, `(x1, y0)` at `o + dx`, `(x0, y1)` at `o + dy`; weights `fx`, `fy`.
#[derive(Debug)]
pub(crate) struct Taps2 {
    o: Vec<u32>,
    dx: Vec<u32>,
    dy: Vec<u32>,
    fx: Vec<f32>,
    fy: Vec<f32>,
    /// Number of valid taps.
    n: usize,
    /// `max(o + dx + dy)` over the valid taps.
    max: usize,
}

impl Taps2 {
    pub fn new(n: usize) -> Self {
        Self {
            n: 0,
            max: 0,
            o: vec![0; n],
            dx: vec![0; n],
            dy: vec![0; n],
            fx: vec![0.0; n],
            fy: vec![0.0; n],
        }
    }
}

/// `floor(v)` as an integer, with `v` first clamped to `[-2, 2^24]` (NaN
/// maps to -2). Exact for every coordinate of an inside pixel (which lies in
/// `[-0.5, W - 0.5]`); other pixels are padding, so their taps only need to
/// stay in bounds after clamping. Branch-free and vectorisable.
#[inline(always)]
// `f32::clamp` would keep NaN, which `to_int_unchecked` must not see.
#[allow(clippy::manual_clamp)]
pub(crate) fn floor_i32(v: f32) -> i32 {
    let c = v.max(-2.0).min(16_777_216.0);
    // SAFETY: `c` is finite and within `[-2, 2^24]`, so it is representable
    // as an `i32` after truncation.
    let t: i32 = unsafe { c.to_int_unchecked() };
    t - ((t as f32) > c) as i32
}

#[inline(always)]
pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
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

/// Chroma planes of a Y'CbCr frame.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Chroma<'a> {
    /// Interleaved U, V (NV12): one channel with `comps == 2`.
    Interleaved(Chan<'a>),
    /// Separate U and V planes of identical dimensions.
    Planar(Chan<'a>, Chan<'a>),
}

impl<'a> Chroma<'a> {
    /// The channel whose geometry (`w`, `h`, `step`) both components share.
    pub fn geometry(&self) -> &Chan<'a> {
        match self {
            Chroma::Interleaved(c) | Chroma::Planar(c, _) => c,
        }
    }
}

/// A validated source frame.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Src<'a> {
    /// Planar or semi-planar Y'CbCr.
    Yuv { y: Chan<'a>, chroma: Chroma<'a>, shift: (u32, u32), k: YuvCoeffs },
    /// Single luma channel.
    Gray(Chan<'a>),
    /// Packed 3-byte pixels (`comps == 3`); `bgr` if the bytes are B, G, R.
    Rgb { c: Chan<'a>, bgr: bool },
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
        let k = YuvCoeffs::new(desc.color.range, desc.color.matrix);
        Ok(match desc.format {
            PixelFormat::Nv12 => {
                check_pitch(0, w)?;
                check_pitch(1, cw * 2)?;
                Src::Yuv {
                    y: Chan::new(data, p[0].offset, p[0].pitch, (1, 1), w, h)?,
                    chroma: Chroma::Interleaved(Chan::new(
                        data,
                        p[1].offset,
                        p[1].pitch,
                        (2, 2),
                        cw,
                        ch,
                    )?),
                    shift: (sx, sy),
                    k,
                }
            }
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                check_pitch(0, w)?;
                check_pitch(1, cw)?;
                check_pitch(2, cw)?;
                Src::Yuv {
                    y: Chan::new(data, p[0].offset, p[0].pitch, (1, 1), w, h)?,
                    chroma: Chroma::Planar(
                        Chan::new(data, p[1].offset, p[1].pitch, (1, 1), cw, ch)?,
                        Chan::new(data, p[2].offset, p[2].pitch, (1, 1), cw, ch)?,
                    ),
                    shift: (sx, sy),
                    k,
                }
            }
            PixelFormat::Gray8 => {
                check_pitch(0, w)?;
                Src::Gray(Chan::new(data, p[0].offset, p[0].pitch, (1, 1), w, h)?)
            }
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
                check_pitch(0, w.saturating_mul(3))?;
                Src::Rgb {
                    c: Chan::new(data, p[0].offset, p[0].pitch, (3, 3), w, h)?,
                    bgr: desc.format == PixelFormat::Bgr24,
                }
            }
        })
    }

    /// Luma (or packed-pixel) channel.
    pub fn main(&self) -> &Chan<'a> {
        match self {
            Src::Yuv { y, .. } | Src::Gray(y) => y,
            Src::Rgb { c, .. } => c,
        }
    }
}
