//! Output stage: normalisation, element type conversion and layout.
//!
//! Input is one output row per channel in pixel units (`0..=255`, output
//! channel order); output is the corresponding bytes of the tensor.

use camtensor_core::{DType, Layout, TensorSpec};
use half::f16;
use half::slice::HalfFloatSliceExt;

/// Per-call encoder configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Encoder {
    pub dtype: DType,
    pub layout: Layout,
    /// Number of channels.
    pub c: usize,
    /// Output width.
    pub w: usize,
    a: [f32; 3],
    b: [f32; 3],
}

/// Reusable temporaries for [`Encoder::encode`].
#[derive(Debug, Default)]
pub(crate) struct EncScratch {
    /// Interleaved row (NHWC).
    inter: Vec<f32>,
    /// Normalised row before f16 conversion.
    norm: Vec<f32>,
    f16s: Vec<f16>,
}

impl Encoder {
    pub fn new(spec: &TensorSpec) -> Self {
        let (a, b) = match spec.dtype {
            DType::U8 => ([1.0; 3], [0.0; 3]),
            _ => spec.normalize.affine_coeffs(),
        };
        Self {
            dtype: spec.dtype,
            layout: spec.layout,
            c: spec.channels(),
            w: spec.width as usize,
            a,
            b,
        }
    }

    /// Number of destination slices a row is written to (one per channel
    /// plane for NCHW, a single interleaved one for NHWC).
    pub fn dst_count(&self) -> usize {
        match self.layout {
            Layout::Nchw => self.c,
            Layout::Nhwc => 1,
        }
    }

    /// Bytes of one row inside one destination slice.
    pub fn row_bytes(&self) -> usize {
        let px = match self.layout {
            Layout::Nchw => 1,
            Layout::Nhwc => self.c,
        };
        self.w * px * self.dtype.size()
    }

    pub fn scratch(&self) -> EncScratch {
        let n = self.w * self.c;
        let f16 = self.dtype == DType::F16;
        EncScratch {
            inter: if self.layout == Layout::Nhwc { vec![0.0; n] } else { Vec::new() },
            norm: if f16 { vec![0.0; n] } else { Vec::new() },
            f16s: if f16 { vec![f16::ZERO; n] } else { Vec::new() },
        }
    }

    /// Encodes one row. `ch[c]` holds at least `w` values for every channel
    /// `c < self.c`; `dst[k]` is exactly `row_bytes()` long for every
    /// `k < dst_count()`.
    pub fn encode(&self, ch: &[Vec<f32>; 3], s: &mut EncScratch, dst: &mut [&mut [u8]]) {
        let w = self.w;
        match self.layout {
            Layout::Nchw => {
                for (c, d) in dst.iter_mut().enumerate().take(self.c) {
                    write(
                        self.dtype,
                        &ch[c][..w],
                        self.a[c],
                        self.b[c],
                        &mut s.norm,
                        &mut s.f16s,
                        d,
                    );
                }
            }
            Layout::Nhwc => {
                let n = w * self.c;
                let tmp = &mut s.inter[..n];
                interleave(ch, self.c, &self.a, &self.b, tmp);
                write(self.dtype, tmp, 1.0, 0.0, &mut s.norm, &mut s.f16s, dst[0]);
            }
        }
    }
}

/// Writes `src * a + b` as `dtype` elements into `dst`.
fn write(
    dtype: DType,
    src: &[f32],
    a: f32,
    b: f32,
    norm: &mut [f32],
    f16s: &mut [f16],
    dst: &mut [u8],
) {
    match dtype {
        DType::F32 => {
            for (d, &v) in dst.chunks_exact_mut(4).zip(src) {
                d.copy_from_slice(&(v * a + b).to_ne_bytes());
            }
        }
        DType::F16 => {
            let n = src.len();
            let t32 = &mut norm[..n];
            for (t, &v) in t32.iter_mut().zip(src) {
                *t = v * a + b;
            }
            let t16 = &mut f16s[..n];
            t16.convert_from_f32_slice(t32);
            for (d, h) in dst.chunks_exact_mut(2).zip(t16.iter()) {
                d.copy_from_slice(&h.to_bits().to_ne_bytes());
            }
        }
        DType::U8 => {
            // floor(v + 0.5) clamped to 0..=255. Clamping first keeps the sum
            // non-negative, so truncation equals floor.
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = ((v * a + b).clamp(0.0, 255.0) + 0.5) as u8;
            }
        }
    }
}

/// `out[x * c + k] = ch[k][x] * a[k] + b[k]`.
fn interleave(ch: &[Vec<f32>; 3], c: usize, a: &[f32; 3], b: &[f32; 3], out: &mut [f32]) {
    match c {
        1 => {
            for (o, &v) in out.iter_mut().zip(&ch[0]) {
                *o = v * a[0] + b[0];
            }
        }
        _ => {
            for (((o, &r), &g), &bl) in out.chunks_exact_mut(3).zip(&ch[0]).zip(&ch[1]).zip(&ch[2])
            {
                o[0] = r * a[0] + b[0];
                o[1] = g * a[1] + b[1];
                o[2] = bl * a[2] + b[2];
            }
        }
    }
}
