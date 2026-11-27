//! Fused colour conversion, resize, normalisation and affine crops (CPU).
//!
//! One pass reads a decoded frame (NV12, YUV420P/422P/444P, GRAY8, RGB24,
//! BGR24) and writes one inference-ready image (`f32`, `f16` or `u8`,
//! NCHW or NHWC, RGB/BGR/GRAY) into a caller-provided buffer.
//!
//! ```
//! use camtensor_core::{ColorSpec, FrameDesc, PixelFormat, TensorSpec};
//!
//! let desc = FrameDesc::packed(1920, 1080, PixelFormat::Nv12, ColorSpec::default());
//! let frame = vec![128u8; desc.byte_len()];
//! let spec = TensorSpec::default(); // 640x640 f32 NCHW RGB letterbox, /255
//! let plan = camtensor_preprocess::plan(1920, 1080, &spec);
//! let mut out = vec![0u8; spec.image_bytes()];
//! camtensor_preprocess::preprocess_into(&desc, &frame, &plan.sampling, &spec, &mut out)?;
//! # Ok::<(), camtensor_core::Error>(())
//! ```
//!
//! # Semantics
//!
//! These are shared bit-for-bit (up to floating-point rounding) with the CUDA
//! kernels in `camtensor-cuda`.
//!
//! * For destination pixel indices `(x, y)` the source position is
//!   `sx = a*x + b*y + c`, `sy = d*x + e*y + f` with the sampling matrix
//!   `[a, b, c, d, e, f]` (see [`Affine2::sampling_matrix`]).
//! * If `sx < -0.5 || sx > W - 0.5 || sy < -0.5 || sy > H - 0.5` the pixel is
//!   the pad colour `spec.pad` (pixel units, output channel order).
//! * Bilinear: taps `floor(s)` and `floor(s) + 1`, each clamped to the plane,
//!   weight `s - floor(s)`. Nearest: `clamp(floor(s + 0.5))`.
//! * Y'CbCr: luma is sampled at `(sx, sy)`, chroma at
//!   `(s + 0.5) / 2^shift - 0.5` on the chroma plane; interpolation happens
//!   in YUV, then BT.601/BT.709, limited/full range conversion to RGB clamped
//!   to `0..=255`. GRAY output from YUV is `clamp(Y')`; from RGB it is
//!   `0.299 R + 0.587 G + 0.114 B`; a GRAY8 source is replicated.
//! * `f32`/`f16`: `v * A[c] + B[c]` with [`Normalize::affine_coeffs`]
//!   (`f16` rounds to nearest even). `u8`: `floor(v + 0.5)` clamped, no
//!   normalisation. Elements are written in native byte order.
//!
//! [`Normalize::affine_coeffs`]: camtensor_core::Normalize::affine_coeffs

mod encode;
mod sample;
mod source;
#[cfg(test)]
mod tests;

use camtensor_core::{Affine2, Error, FrameDesc, Result, Roi, TensorSpec};

use crate::encode::Encoder;
use crate::sample::{SampleScratch, Sampler, Separable};
use crate::source::Src;

/// Geometry of one preprocessing operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Continuous `dst_from_src` transform (map detections back with its
    /// inverse).
    pub dst_from_src: Affine2,
    /// Destination index → source index matrix consumed by
    /// [`preprocess_into`].
    pub sampling: Affine2,
}

impl Plan {
    /// Builds a plan from a continuous `dst_from_src` transform; `None` if it
    /// is singular.
    pub fn from_dst_from_src(dst_from_src: Affine2) -> Option<Self> {
        Some(Self { dst_from_src, sampling: Affine2::sampling_matrix(&dst_from_src)? })
    }
}

/// Plans a full-frame output:
/// `dst_from_src = Affine2::fit(src_w, src_h, spec.width, spec.height, spec.resize)`.
///
/// For degenerate sizes (zero width/height) the sampling matrix falls back
/// to the identity; [`preprocess_into`] rejects such inputs anyway.
pub fn plan(src_w: u32, src_h: u32, spec: &TensorSpec) -> Plan {
    let dst_from_src = Affine2::fit(
        src_w as f32,
        src_h as f32,
        spec.width as f32,
        spec.height as f32,
        spec.resize,
    );
    Plan::from_dst_from_src(dst_from_src)
        .unwrap_or(Plan { dst_from_src, sampling: Affine2::IDENTITY })
}

/// Plans a crop (box, affine warp or keypoint alignment).
pub fn plan_roi(roi: &Roi, spec: &TensorSpec) -> Result<Plan> {
    roi.dst_from_src(spec.width, spec.height, spec.resize)
        .and_then(Plan::from_dst_from_src)
        .ok_or_else(|| Error::Preprocess(format!("degenerate region of interest: {roi:?}")))
}

/// Threading strategy for [`preprocess_into_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Parallelism {
    /// Run on the calling thread (the engine already runs one thread per
    /// camera).
    #[default]
    Single,
    /// Split large outputs into row bands processed on the global rayon
    /// pool. Small outputs, or builds without the `rayon` feature, run on the
    /// calling thread.
    Rayon,
}

/// Outputs with fewer pixels than this are never split across threads.
#[cfg(feature = "rayon")]
const PAR_MIN_PIXELS: usize = 128 * 128;

/// Writes ONE preprocessed image (`spec.image_bytes()` bytes) into `out` on
/// the calling thread. See the crate docs for the exact semantics.
pub fn preprocess_into(
    src: &FrameDesc,
    data: &[u8],
    sampling: &Affine2,
    spec: &TensorSpec,
    out: &mut [u8],
) -> Result<()> {
    preprocess_into_with(src, data, sampling, spec, out, Parallelism::Single)
}

/// [`preprocess_into`] with explicit [`Parallelism`].
pub fn preprocess_into_with(
    src: &FrameDesc,
    data: &[u8],
    sampling: &Affine2,
    spec: &TensorSpec,
    out: &mut [u8],
    parallelism: Parallelism,
) -> Result<()> {
    spec.validate().map_err(|e| Error::Preprocess(e.to_string()))?;
    if out.len() != spec.image_bytes() {
        return Err(Error::Preprocess(format!(
            "output buffer is {} bytes, expected {}",
            out.len(),
            spec.image_bytes()
        )));
    }
    if !sampling.0.iter().all(|v| v.is_finite()) {
        return Err(Error::Preprocess(format!("non-finite sampling matrix {sampling:?}")));
    }
    let source = Src::new(src, data)?;
    let m = sampling.0;
    let out_w = spec.width as usize;
    let out_h = spec.height as usize;
    let sep = if m[1] == 0.0 && m[3] == 0.0 {
        Separable::new(&m, out_w, &source, spec.interp)
    } else {
        None
    };
    let pad = match spec.color {
        camtensor_core::ColorOrder::Gray => [spec.pad[0]; 3],
        _ => spec.pad,
    };
    let sampler =
        Sampler { src: source, m, out_w, interp: spec.interp, color: spec.color, pad, sep };
    let enc = Encoder::new(spec);
    let plane_bytes = enc.row_bytes() * out_h;
    let mut dsts: [&mut [u8]; 3] = [&mut [], &mut [], &mut []];
    for (d, chunk) in dsts.iter_mut().zip(out.chunks_exact_mut(plane_bytes)) {
        *d = chunk;
    }
    let dsts = &mut dsts[..enc.dst_count()];

    match parallelism {
        #[cfg(feature = "rayon")]
        Parallelism::Rayon if out_w * out_h >= PAR_MIN_PIXELS => {
            run_parallel(&sampler, &enc, out_h, dsts);
        }
        _ => run_band(&sampler, &enc, 0, out_h, dsts),
    }
    Ok(())
}

/// Processes output rows `y0..y0 + rows`; `dsts[k]` holds exactly those rows.
fn run_band(sampler: &Sampler<'_>, enc: &Encoder, y0: usize, rows: usize, dsts: &mut [&mut [u8]]) {
    let rb = enc.row_bytes();
    let nd = dsts.len();
    let mut ss = SampleScratch::new(sampler.out_w);
    let mut es = enc.scratch();
    let mut pad_cache: Option<[Vec<u8>; 3]> = None;
    let mut iters: Vec<_> = dsts.iter_mut().map(|d| d.chunks_exact_mut(rb)).collect();
    for y in y0..y0 + rows {
        let mut row: [&mut [u8]; 3] = [&mut [], &mut [], &mut []];
        for (r, it) in row.iter_mut().zip(iters.iter_mut()) {
            *r = it.next().expect("band buffer holds `rows` rows");
        }
        let row = &mut row[..nd];
        if sampler.row(y, &mut ss) {
            enc.encode(&ss.ch, &mut es, row);
        } else if let Some(cache) = &pad_cache {
            for (r, c) in row.iter_mut().zip(cache) {
                r.copy_from_slice(c);
            }
        } else {
            sampler.fill_pad(&mut ss);
            enc.encode(&ss.ch, &mut es, row);
            let mut cache: [Vec<u8>; 3] = Default::default();
            for (c, r) in cache.iter_mut().zip(row.iter()) {
                c.extend_from_slice(r);
            }
            pad_cache = Some(cache);
        }
    }
}

#[cfg(feature = "rayon")]
fn run_parallel(sampler: &Sampler<'_>, enc: &Encoder, out_h: usize, dsts: &mut [&mut [u8]]) {
    use rayon::prelude::*;

    let threads = rayon::current_num_threads();
    if threads <= 1 {
        return run_band(sampler, enc, 0, out_h, dsts);
    }
    let band = out_h.div_ceil(threads * 4).max(8);
    let rb = enc.row_bytes();
    let nd = dsts.len();
    let mut chunks: Vec<_> = dsts.iter_mut().map(|d| d.chunks_mut(band * rb)).collect();
    let mut bands = Vec::with_capacity(out_h.div_ceil(band));
    for y0 in (0..out_h).step_by(band) {
        let mut b: [&mut [u8]; 3] = [&mut [], &mut [], &mut []];
        for (s, it) in b.iter_mut().zip(chunks.iter_mut()) {
            *s = it.next().expect("one chunk per band");
        }
        bands.push((y0, b));
    }
    bands.into_par_iter().for_each(|(y0, mut b)| {
        let rows = b[0].len() / rb;
        run_band(sampler, enc, y0, rows, &mut b[..nd]);
    });
}
