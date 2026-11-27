//! Sampling stage: produces one output row per channel in pixel units.
//!
//! Two paths share the colour conversion and the encoder:
//!
//! * **separable** (`b == 0 && d == 0`: every resize mode and axis-aligned
//!   box crops). Horizontal taps are precomputed once per call, vertical taps
//!   once per row, and the inside region of a row is a contiguous column
//!   range so padding is filled without sampling. Bilinear rows are produced
//!   in two passes: a contiguous (vectorised) vertical blend of the two
//!   source rows into an `f32` line, then a 2-tap horizontal lerp from it.
//! * **general** (rotations / similarity crops): taps are computed per
//!   pixel, with the source-kind and filter dispatch hoisted out of the loop.

use camtensor_core::{ColorOrder, Interp};

use crate::source::{Chan, Chroma, Src, Taps2, YuvCoeffs, floor_i32, lerp};

/// Destination-pixel outside test on one axis (see the crate docs).
#[inline(always)]
pub(crate) fn outside(s: f32, n: usize) -> bool {
    (s < -0.5) | (s > n as f32 - 0.5)
}

/// `1 / 2^shift` (exact).
#[inline]
fn inv_scale(shift: u32) -> f32 {
    1.0 / (1u32 << shift) as f32
}

/// Chroma coordinate for a luma coordinate: `(s + 0.5) / 2^shift - 0.5`.
/// `inv` is `1 / 2^shift`; multiplying by an exact power-of-two reciprocal
/// gives bit-identical results to the division.
#[inline(always)]
fn chroma_coord(s: f32, inv: f32) -> f32 {
    (s + 0.5) * inv - 0.5
}

/// Vertical taps: source rows `r0`, `r1` (clamped) and weight `f` of `r1`.
#[derive(Debug, Clone, Copy)]
struct RowTap {
    r0: usize,
    r1: usize,
    f: f32,
}

impl RowTap {
    #[inline]
    fn new(interp: Interp, s: f32, h: usize) -> Self {
        let max = h as i32 - 1;
        match interp {
            Interp::Bilinear => {
                let y0 = floor_i32(s);
                RowTap {
                    r0: y0.clamp(0, max) as usize,
                    r1: y0.wrapping_add(1).clamp(0, max) as usize,
                    f: s - y0 as f32,
                }
            }
            Interp::Nearest => {
                let r = floor_i32(s + 0.5).clamp(0, max) as usize;
                RowTap { r0: r, r1: r, f: 0.0 }
            }
        }
    }
}

/// One horizontal tap: blends samples at relative byte indices `i` and
/// `i + ColTaps::d` with weight `f` of the second.
#[derive(Debug, Clone, Copy)]
struct ColTap {
    i: u32,
    f: f32,
}

/// Horizontal taps of a contiguous range of output columns on one channel.
///
/// Indices are relative to `base` (a byte offset into a channel row).
/// Clamping is folded into the taps so the second tap is always `i + d`:
/// a first tap left of the plane becomes `(0, f = 0)`, one at or past the
/// last sample becomes `(w - 2, f = 1)` â€” both yield exactly the clamped
/// edge sample.
///
/// Invariant: `i <= max` for every tap and
/// `base + max + d + comps <= base + span <= chan.row_len()`.
#[derive(Debug)]
struct ColTaps {
    taps: Vec<ColTap>,
    base: usize,
    d: usize,
    max: usize,
    /// Bytes of a row (from `base`) the taps read: `max + d + comps`.
    span: usize,
}

impl ColTaps {
    fn new(interp: Interp, coords: impl Iterator<Item = f32>, c: &Chan<'_>) -> Self {
        let (w, step) = (c.w, c.step);
        let abs: Vec<(usize, f32)> = coords
            .map(|s| match interp {
                Interp::Nearest => (floor_i32(s + 0.5).clamp(0, w as i32 - 1) as usize * step, 0.0),
                Interp::Bilinear => {
                    let x0 = floor_i32(s);
                    if w == 1 || x0 < 0 {
                        (0, 0.0)
                    } else if x0 as usize >= w - 1 {
                        ((w - 2) * step, 1.0)
                    } else {
                        (x0 as usize * step, s - x0 as f32)
                    }
                }
            })
            .collect();
        let base = abs.iter().map(|t| t.0).min().unwrap_or(0);
        let max = abs.iter().map(|t| t.0 - base).max().unwrap_or(0);
        let d = if interp == Interp::Bilinear && w >= 2 { step } else { 0 };
        let taps = abs.iter().map(|&(i, f)| ColTap { i: (i - base) as u32, f }).collect();
        let span = max + d + c.comps;
        debug_assert!(base + span <= c.row_len());
        Self { taps, base, d, max, span }
    }

    /// The bytes of row `y` of `c` the taps read.
    #[inline]
    fn row<'a>(&self, c: &Chan<'a>, y: usize) -> &'a [u8] {
        &c.row(y)[self.base..self.base + self.span]
    }
}

/// Precomputed state of the separable path.
#[derive(Debug)]
pub(crate) struct Separable {
    /// Inside column range `[xb, xe)`.
    xb: usize,
    xe: usize,
    /// Taps on the main channel (luma / gray / packed RGB).
    main: ColTaps,
    /// Taps on the chroma channels (YUV only).
    chroma: Option<ColTaps>,
}

impl Separable {
    /// Builds the tables. Returns `None` if the inside columns are not a
    /// contiguous range (cannot happen for a linear map, kept as a guard).
    pub fn new(m: &[f32; 6], out_w: usize, src: &Src<'_>, interp: Interp) -> Option<Self> {
        let w = src.main().w;
        let sx: Vec<f32> = (0..out_w).map(|x| m[0] * x as f32 + m[2]).collect();
        let inside: Vec<bool> = sx.iter().map(|&s| !outside(s, w)).collect();
        let xb = inside.iter().position(|&i| i).unwrap_or(out_w);
        let xe = inside.iter().rposition(|&i| i).map_or(xb, |p| p + 1);
        if inside[xb..xe].iter().any(|&i| !i) {
            return None;
        }
        let cols = &sx[xb..xe];
        let main = ColTaps::new(interp, cols.iter().copied(), src.main());
        let chroma = match src {
            Src::Yuv { chroma, shift, .. } => Some(ColTaps::new(
                interp,
                cols.iter().map(|&s| chroma_coord(s, inv_scale(shift.0))),
                chroma.geometry(),
            )),
            _ => None,
        };
        Some(Self { xb, xe, main, chroma })
    }
}

/// Per-thread temporaries of the sampling stage.
#[derive(Debug)]
pub(crate) struct SampleScratch {
    /// Raw interpolated source components (Y, U, V or packed byte order).
    planes: [Vec<f32>; 3],
    /// Output channels in pixel units, output order.
    pub ch: [Vec<f32>; 3],
    /// Vertically blended source lines (separable bilinear).
    lines: [Vec<f32>; 2],
    /// Source coordinates, chroma coordinates, outside flags and taps
    /// (general path).
    sx: Vec<f32>,
    sy: Vec<f32>,
    cx: Vec<f32>,
    cy: Vec<f32>,
    mask: Vec<bool>,
    taps: Taps2,
}

impl SampleScratch {
    pub fn new(out_w: usize) -> Self {
        let v = || vec![0.0f32; out_w];
        Self {
            planes: [v(), v(), v()],
            ch: [v(), v(), v()],
            lines: [Vec::new(), Vec::new()],
            sx: v(),
            sy: v(),
            cx: v(),
            cy: v(),
            mask: vec![false; out_w],
            taps: Taps2::new(out_w),
        }
    }
}

/// Everything the sampler needs for one call.
#[derive(Debug)]
pub(crate) struct Sampler<'a> {
    pub src: Src<'a>,
    pub m: [f32; 6],
    pub out_w: usize,
    pub interp: Interp,
    pub color: ColorOrder,
    pub pad: [f32; 3],
    pub sep: Option<Separable>,
}

impl Sampler<'_> {
    /// Fills `s.ch` for output row `y`. Returns `false` if the whole row is
    /// padding (in which case `s.ch` is left untouched).
    pub fn row(&self, y: usize, s: &mut SampleScratch) -> bool {
        match &self.sep {
            Some(sep) => self.row_separable(sep, y, s),
            None => {
                match self.interp {
                    Interp::Bilinear => self.row_general::<true>(y, s),
                    Interp::Nearest => self.row_general::<false>(y, s),
                }
                true
            }
        }
    }

    /// Fills `s.ch[..][..out_w]` with the padding colour.
    pub fn fill_pad(&self, s: &mut SampleScratch) {
        let w = self.out_w;
        for (c, ch) in s.ch.iter_mut().enumerate().take(self.color.channels()) {
            ch[..w].fill(self.pad[c]);
        }
    }

    fn row_separable(&self, sep: &Separable, y: usize, s: &mut SampleScratch) -> bool {
        let main = self.src.main();
        let sy = self.m[4] * y as f32 + self.m[5];
        if outside(sy, main.h) || sep.xb == sep.xe {
            return false;
        }
        let n = sep.xe - sep.xb;
        let interp = self.interp;
        let [p0, p1, p2] = &mut s.planes;
        let [l0, l1] = &mut s.lines;
        let (p0, p1, p2) = (&mut p0[..n], &mut p1[..n], &mut p2[..n]);
        let t = RowTap::new(interp, sy, main.h);
        match &self.src {
            Src::Yuv { y: yc, chroma, shift, .. } => {
                sample_cols::<1>(interp, yc, &sep.main, t, l0, [p0]);
                let ct = sep.chroma.as_ref().expect("yuv source has chroma taps");
                let g = chroma.geometry();
                let rt = RowTap::new(interp, chroma_coord(sy, inv_scale(shift.1)), g.h);
                match chroma {
                    Chroma::Interleaved(c) => sample_cols::<2>(interp, c, ct, rt, l1, [p1, p2]),
                    Chroma::Planar(u, v) => {
                        sample_cols::<1>(interp, u, ct, rt, l1, [p1]);
                        sample_cols::<1>(interp, v, ct, rt, l1, [p2]);
                    }
                }
            }
            Src::Gray(c) => sample_cols::<1>(interp, c, &sep.main, t, l0, [p0]),
            Src::Rgb { c, .. } => sample_cols::<3>(interp, c, &sep.main, t, l0, [p0, p1, p2]),
        }
        self.convert(&s.planes, &mut s.ch, sep.xb, n);
        for (c, ch) in s.ch.iter_mut().enumerate().take(self.color.channels()) {
            ch[..sep.xb].fill(self.pad[c]);
            ch[sep.xe..self.out_w].fill(self.pad[c]);
        }
        true
    }

    fn row_general<const BIL: bool>(&self, y: usize, s: &mut SampleScratch) {
        let w = self.out_w;
        let main = self.src.main();
        let m = &self.m;
        let yf = y as f32;
        let (by, ey) = (m[1] * yf, m[4] * yf);
        // Coordinates and the outside test (vectorised). Same evaluation
        // order as `a*x + b*y + c`.
        // `x` goes through i32 (out_w <= 16384) so the conversion vectorises.
        let (sxs, sys, mask) = (&mut s.sx[..w], &mut s.sy[..w], &mut s.mask[..w]);
        for x in 0..w {
            let xf = x as i32 as f32;
            let (sx, sy) = (m[0] * xf + by + m[2], m[3] * xf + ey + m[5]);
            sxs[x] = sx;
            sys[x] = sy;
            mask[x] = outside(sx, main.w) | outside(sy, main.h);
        }
        // Outside pixels are sampled too (their taps are clamped, so this is
        // harmless) and overwritten with padding below; this keeps the loops
        // branch-free.
        let [p0, p1, p2] = &mut s.planes;
        let (p0, p1, p2) = (&mut p0[..w], &mut p1[..w], &mut p2[..w]);
        let t = &mut s.taps;
        main.taps(BIL, &s.sx[..w], &s.sy[..w], t);
        match &self.src {
            Src::Yuv { y: yc, chroma, shift, .. } => {
                yc.gather::<1>(BIL, t, [p0]);
                let (ix, iy) = (inv_scale(shift.0), inv_scale(shift.1));
                for (c, &l) in s.cx[..w].iter_mut().zip(&s.sx[..w]) {
                    *c = chroma_coord(l, ix);
                }
                for (c, &l) in s.cy[..w].iter_mut().zip(&s.sy[..w]) {
                    *c = chroma_coord(l, iy);
                }
                chroma.geometry().taps(BIL, &s.cx[..w], &s.cy[..w], t);
                match chroma {
                    Chroma::Interleaved(c) => c.gather::<2>(BIL, t, [p1, p2]),
                    Chroma::Planar(u, v) => {
                        u.gather::<1>(BIL, t, [p1]);
                        // Taps embed the pitch, which may differ per plane.
                        if v.pitch != u.pitch {
                            v.taps(BIL, &s.cx[..w], &s.cy[..w], t);
                        }
                        v.gather::<1>(BIL, t, [p2]);
                    }
                }
            }
            Src::Gray(c) => c.gather::<1>(BIL, t, [p0]),
            Src::Rgb { c, .. } => c.gather::<3>(BIL, t, [p0, p1, p2]),
        }
        self.convert(&s.planes, &mut s.ch, 0, w);
        let mask = &s.mask[..w];
        for (c, ch) in s.ch.iter_mut().enumerate().take(self.color.channels()) {
            let pad = self.pad[c];
            for (v, &o) in ch[..w].iter_mut().zip(mask) {
                if o {
                    *v = pad;
                }
            }
        }
    }

    /// Converts `planes[..][..n]` into `ch[..][x0..x0 + n]` (output order).
    fn convert(&self, planes: &[Vec<f32>; 3], ch: &mut [Vec<f32>; 3], x0: usize, n: usize) {
        let [c0, c1, c2] = ch;
        let (c0, c1, c2) = (&mut c0[x0..x0 + n], &mut c1[x0..x0 + n], &mut c2[x0..x0 + n]);
        let (p0, p1, p2) = (&planes[0][..n], &planes[1][..n], &planes[2][..n]);
        match (&self.src, self.color) {
            (Src::Yuv { k, .. }, ColorOrder::Rgb) => yuv_to_rgb(k, p0, p1, p2, c0, c1, c2),
            (Src::Yuv { k, .. }, ColorOrder::Bgr) => yuv_to_rgb(k, p0, p1, p2, c2, c1, c0),
            (Src::Yuv { k, .. }, ColorOrder::Gray) => {
                for (o, &y) in c0.iter_mut().zip(p0) {
                    *o = ((y - k.y_off) * k.y_scale).clamp(0.0, 255.0);
                }
            }
            (Src::Gray(_), ColorOrder::Gray) => c0.copy_from_slice(p0),
            (Src::Gray(_), _) => {
                c0.copy_from_slice(p0);
                c1.copy_from_slice(p0);
                c2.copy_from_slice(p0);
            }
            (Src::Rgb { bgr, .. }, order) => {
                // `planes` are in byte order; resolve to R, G, B first.
                let (r, g, b) = if *bgr { (p2, p1, p0) } else { (p0, p1, p2) };
                match order {
                    ColorOrder::Rgb => {
                        c0.copy_from_slice(r);
                        c1.copy_from_slice(g);
                        c2.copy_from_slice(b);
                    }
                    ColorOrder::Bgr => {
                        c0.copy_from_slice(b);
                        c1.copy_from_slice(g);
                        c2.copy_from_slice(r);
                    }
                    ColorOrder::Gray => {
                        for (((o, &r), &g), &b) in c0.iter_mut().zip(r).zip(g).zip(b) {
                            *o = 0.299 * r + 0.587 * g + 0.114 * b;
                        }
                    }
                }
            }
        }
    }
}

fn yuv_to_rgb(
    k: &YuvCoeffs,
    y: &[f32],
    u: &[f32],
    v: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let n = y.len();
    let (u, v, r, g, b) = (&u[..n], &v[..n], &mut r[..n], &mut g[..n], &mut b[..n]);
    for i in 0..n {
        let yp = (y[i] - k.y_off) * k.y_scale;
        let cb = (u[i] - 128.0) * k.c_scale;
        let cr = (v[i] - 128.0) * k.c_scale;
        r[i] = (yp + k.rv * cr).clamp(0.0, 255.0);
        g[i] = (yp - k.gu * cb - k.gv * cr).clamp(0.0, 255.0);
        b[i] = (yp + k.bu * cb).clamp(0.0, 255.0);
    }
}

/// Samples the first `N` components of `c` for every column of `t` on the
/// source rows given by `rt`, writing `out[k][i]`.
#[inline(always)]
fn sample_cols<const N: usize>(
    interp: Interp,
    c: &Chan<'_>,
    t: &ColTaps,
    rt: RowTap,
    line: &mut Vec<f32>,
    out: [&mut [f32]; N],
) {
    assert!(N <= c.comps);
    match interp {
        Interp::Nearest => gather_nearest::<N>(t.row(c, rt.r0), t, out),
        Interp::Bilinear => {
            // Blending whole rows first is cheaper unless most of the row is
            // skipped (strong downscale), then gather the four taps directly.
            if t.span * DIRECT_RATIO_DEN > t.taps.len() * c.comps * DIRECT_RATIO_NUM {
                gather_bilinear::<N>(t.row(c, rt.r0), t.row(c, rt.r1), rt.f, t, out);
            } else {
                if line.len() < t.span {
                    line.resize(t.span, 0.0);
                }
                let line = &mut line[..t.span];
                vblend(t.row(c, rt.r0), t.row(c, rt.r1), rt.f, line);
                hlerp::<N>(line, t, out);
            }
        }
    }
}

/// Use the direct 4-tap gather when `span / (columns * comps)` exceeds
/// `NUM / DEN`.
const DIRECT_RATIO_NUM: usize = 4;
const DIRECT_RATIO_DEN: usize = 1;

/// `out[k][i] = bilinear(r0, r1)` at taps `t[i]`, reading bytes directly.
#[inline(always)]
fn gather_bilinear<const N: usize>(
    r0: &[u8],
    r1: &[u8],
    fy: f32,
    t: &ColTaps,
    out: [&mut [f32]; N],
) {
    let n = t.taps.len();
    let d = t.d;
    assert!(t.max + d + N <= r0.len() && t.max + d + N <= r1.len());
    for o in &out {
        assert!(o.len() >= n);
    }
    let mut out = out;
    for (i, tap) in t.taps.iter().enumerate() {
        let a = tap.i as usize;
        for (k, o) in out.iter_mut().enumerate() {
            // SAFETY: `a <= t.max` (ColTaps invariant) and `k < N`, so both
            // `a + k` and `a + d + k` are below `t.max + d + N`, which is at
            // most the length of both rows (asserted above).
            let (p00, p01, p10, p11) = unsafe {
                (
                    *r0.get_unchecked(a + k),
                    *r0.get_unchecked(a + d + k),
                    *r1.get_unchecked(a + k),
                    *r1.get_unchecked(a + d + k),
                )
            };
            let top = lerp(p00 as f32, p01 as f32, tap.f);
            let bot = lerp(p10 as f32, p11 as f32, tap.f);
            o[i] = lerp(top, bot, fy);
        }
    }
}

/// `out[j] = lerp(r0[j], r1[j], f)` over contiguous bytes (vectorised).
#[inline(always)]
fn vblend(r0: &[u8], r1: &[u8], f: f32, out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(r0).zip(r1) {
        let a = a as f32;
        *o = a + f * (b as f32 - a);
    }
}

/// `out[k][i] = lerp(line[t_i + k], line[t_i + d + k], f_i)`.
#[inline(always)]
fn hlerp<const N: usize>(line: &[f32], t: &ColTaps, out: [&mut [f32]; N]) {
    let n = t.taps.len();
    let d = t.d;
    // Establishes the bound the unchecked loads rely on: the largest index
    // read is `max + d + N - 1 < span == line.len()`.
    assert!(t.max + d + N <= line.len());
    for o in &out {
        assert!(o.len() >= n);
    }
    let mut out = out;
    for (i, tap) in t.taps.iter().enumerate() {
        let a = tap.i as usize;
        for (k, o) in out.iter_mut().enumerate() {
            // SAFETY: `a <= t.max` (ColTaps invariant) and `k < N`, so both
            // indices are below `t.max + d + N <= line.len()` (asserted).
            let (p0, p1) = unsafe { (*line.get_unchecked(a + k), *line.get_unchecked(a + d + k)) };
            o[i] = lerp(p0, p1, tap.f);
        }
    }
}

/// `out[k][i] = row[t_i + k]` (nearest neighbour).
#[inline(always)]
fn gather_nearest<const N: usize>(row: &[u8], t: &ColTaps, out: [&mut [f32]; N]) {
    let n = t.taps.len();
    assert!(t.max + N <= row.len());
    for o in &out {
        assert!(o.len() >= n);
    }
    let mut out = out;
    for (i, tap) in t.taps.iter().enumerate() {
        let a = tap.i as usize;
        for (k, o) in out.iter_mut().enumerate() {
            // SAFETY: `a <= t.max` and `k < N`, so `a + k < t.max + N <=
            // row.len()` (asserted above).
            o[i] = unsafe { *row.get_unchecked(a + k) } as f32;
        }
    }
}
