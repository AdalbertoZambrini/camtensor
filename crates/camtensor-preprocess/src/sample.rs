//! Sampling stage: produces one output row per channel in pixel units.
//!
//! Two paths share the colour conversion and the encoder:
//!
//! * **separable** (`b == 0 && d == 0`, every resize mode and axis-aligned
//!   box crop): horizontal taps are precomputed once per call and vertical
//!   taps once per row, the inside region of a row is a contiguous column
//!   range and padding is filled without sampling.
//! * **general** (rotations / similarity crops): taps are computed per pixel.

use camtensor_core::{ColorOrder, Interp};

use crate::source::{Chan, Src, YuvCoeffs};

/// `floor(v)` as an integer. Exact for `|v| < 2^31`; NaN maps to 0.
#[inline(always)]
pub(crate) fn floor_i32(v: f32) -> i32 {
    let t = v as i32;
    t.wrapping_sub(((t as f32) > v) as i32)
}

/// Destination-pixel outside test on one axis (see the crate docs).
#[inline(always)]
pub(crate) fn outside(s: f32, n: usize) -> bool {
    s < -0.5 || s > n as f32 - 0.5
}

/// Chroma coordinate for a luma coordinate (`(s + 0.5) / 2^shift - 0.5`).
#[inline(always)]
fn chroma_coord(s: f32, shift: u32) -> f32 {
    (s + 0.5) / (1u32 << shift) as f32 - 0.5
}

/// One filter tap pair along an axis: sample indices `i0`, `i1` (already
/// clamped and multiplied by the channel step) and weight `f` of `i1`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Tap {
    i0: u32,
    i1: u32,
    f: f32,
}

impl Tap {
    /// Bilinear tap for coordinate `s` on an axis of `n` samples.
    #[inline(always)]
    fn bilinear(s: f32, n: usize, step: usize) -> Tap {
        let x0 = floor_i32(s);
        let f = s - x0 as f32;
        let max = n as i32 - 1;
        let i0 = x0.clamp(0, max) as usize * step;
        let i1 = x0.wrapping_add(1).clamp(0, max) as usize * step;
        Tap { i0: i0 as u32, i1: i1 as u32, f }
    }

    /// Nearest-neighbour tap (`i1 == i0`, `f == 0`).
    #[inline(always)]
    fn nearest(s: f32, n: usize, step: usize) -> Tap {
        let max = n as i32 - 1;
        let i = (floor_i32(s + 0.5).clamp(0, max) as usize * step) as u32;
        Tap { i0: i, i1: i, f: 0.0 }
    }

    #[inline(always)]
    fn new(interp: Interp, s: f32, n: usize, step: usize) -> Tap {
        match interp {
            Interp::Bilinear => Tap::bilinear(s, n, step),
            Interp::Nearest => Tap::nearest(s, n, step),
        }
    }
}

/// Horizontal taps for a contiguous range of output columns.
///
/// Invariant: every `i0`/`i1` is `<= max`, and `max < chan.row_len()` for the
/// channel the table was built for (indices are clamped to `w - 1` and
/// multiplied by `step`).
#[derive(Debug)]
struct ColTaps {
    taps: Vec<Tap>,
    max: usize,
}

impl ColTaps {
    fn new(interp: Interp, coords: impl Iterator<Item = f32>, c: &Chan<'_>) -> Self {
        let taps: Vec<Tap> = coords.map(|s| Tap::new(interp, s, c.w, c.step)).collect();
        let max = taps.iter().map(|t| t.i0.max(t.i1) as usize).max().unwrap_or(0);
        debug_assert!(max < c.row_len());
        Self { taps, max }
    }
}

/// Precomputed state of the separable path.
#[derive(Debug)]
pub(crate) struct Separable {
    /// Inside column range `[xb, xe)`.
    xb: usize,
    xe: usize,
    /// Taps on the first channel (luma / gray / RGB).
    luma: ColTaps,
    /// Taps on the chroma channels (YUV only).
    chroma: Option<ColTaps>,
}

impl Separable {
    /// Builds the tables. Returns `None` if the inside columns are not a
    /// contiguous range (cannot happen for a linear map, kept as a guard).
    pub fn new(m: &[f32; 6], out_w: usize, src: &Src<'_>, interp: Interp) -> Option<Self> {
        let (w, _) = src.dims();
        let sx: Vec<f32> = (0..out_w).map(|x| m[0] * x as f32 + m[2]).collect();
        let inside: Vec<bool> = sx.iter().map(|&s| !outside(s, w)).collect();
        let xb = inside.iter().position(|&i| i).unwrap_or(out_w);
        let xe = inside.iter().rposition(|&i| i).map_or(xb, |p| p + 1);
        if inside[xb..xe].iter().any(|&i| !i) {
            return None;
        }
        let cols = &sx[xb..xe];
        let (luma, chroma) = match src {
            Src::Yuv { y, u, shift, .. } => (
                ColTaps::new(interp, cols.iter().copied(), y),
                Some(ColTaps::new(interp, cols.iter().map(|&s| chroma_coord(s, shift.0)), u)),
            ),
            Src::Gray(c) => (ColTaps::new(interp, cols.iter().copied(), c), None),
            Src::Rgb(c) => (ColTaps::new(interp, cols.iter().copied(), &c[0]), None),
        };
        Some(Self { xb, xe, luma, chroma })
    }
}

/// Per-thread temporaries of the sampling stage.
#[derive(Debug)]
pub(crate) struct SampleScratch {
    /// Raw interpolated source channels (Y, U, V or R, G, B).
    planes: [Vec<f32>; 3],
    /// Output channels in pixel units, output order.
    pub ch: [Vec<f32>; 3],
    /// Per-pixel outside flags (general path).
    mask: Vec<bool>,
}

impl SampleScratch {
    pub fn new(out_w: usize) -> Self {
        let v = || vec![0.0f32; out_w];
        Self { planes: [v(), v(), v()], ch: [v(), v(), v()], mask: vec![false; out_w] }
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
                self.row_general(y, s);
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
        let (_, h) = self.src.dims();
        let sy = self.m[4] * y as f32 + self.m[5];
        if outside(sy, h) || sep.xb == sep.xe {
            return false;
        }
        let n = sep.xe - sep.xb;
        let interp = self.interp;
        let [p0, p1, p2] = &mut s.planes;
        match &self.src {
            Src::Yuv { y: yc, u, v, shift, .. } => {
                let t = Tap::new(interp, sy, yc.h, 1);
                gather::<1>(
                    interp,
                    [yc.row(t.i0 as usize)],
                    [yc.row(t.i1 as usize)],
                    t.f,
                    &sep.luma,
                    [&mut p0[..n]],
                );
                let ct = Tap::new(interp, chroma_coord(sy, shift.1), u.h, 1);
                let chroma = sep.chroma.as_ref().expect("yuv source has chroma taps");
                gather::<2>(
                    interp,
                    [u.row(ct.i0 as usize), v.row(ct.i0 as usize)],
                    [u.row(ct.i1 as usize), v.row(ct.i1 as usize)],
                    ct.f,
                    chroma,
                    [&mut p1[..n], &mut p2[..n]],
                );
            }
            Src::Gray(c) => {
                let t = Tap::new(interp, sy, c.h, 1);
                gather::<1>(
                    interp,
                    [c.row(t.i0 as usize)],
                    [c.row(t.i1 as usize)],
                    t.f,
                    &sep.luma,
                    [&mut p0[..n]],
                );
            }
            Src::Rgb(c) => {
                let t = Tap::new(interp, sy, c[0].h, 1);
                let (r0, r1) = (t.i0 as usize, t.i1 as usize);
                gather::<3>(
                    interp,
                    [c[0].row(r0), c[1].row(r0), c[2].row(r0)],
                    [c[0].row(r1), c[1].row(r1), c[2].row(r1)],
                    t.f,
                    &sep.luma,
                    [&mut p0[..n], &mut p1[..n], &mut p2[..n]],
                );
            }
        }
        self.convert(&s.planes, &mut s.ch, sep.xb, n);
        let nc = self.color.channels();
        for (c, ch) in s.ch.iter_mut().enumerate().take(nc) {
            ch[..sep.xb].fill(self.pad[c]);
            ch[sep.xe..self.out_w].fill(self.pad[c]);
        }
        true
    }

    fn row_general(&self, y: usize, s: &mut SampleScratch) {
        let w = self.out_w;
        let (sw, sh) = self.src.dims();
        let m = &self.m;
        let yf = y as f32;
        let interp = self.interp;
        let [p0, p1, p2] = &mut s.planes;
        for x in 0..w {
            let xf = x as f32;
            let sx = m[0] * xf + m[1] * yf + m[2];
            let sy = m[3] * xf + m[4] * yf + m[5];
            let out = outside(sx, sw) || outside(sy, sh);
            s.mask[x] = out;
            if out {
                p0[x] = 0.0;
                p1[x] = 0.0;
                p2[x] = 0.0;
                continue;
            }
            match &self.src {
                Src::Yuv { y: yc, u, v, shift, .. } => {
                    p0[x] = sample2d(interp, yc, sx, sy);
                    let (cx, cy) = (chroma_coord(sx, shift.0), chroma_coord(sy, shift.1));
                    let tx = Tap::new(interp, cx, u.w, u.step);
                    let ty = Tap::new(interp, cy, u.h, 1);
                    p1[x] = sample_taps(interp, u, tx, ty);
                    p2[x] = sample_taps(interp, v, tx, ty);
                }
                Src::Gray(c) => p0[x] = sample2d(interp, c, sx, sy),
                Src::Rgb(c) => {
                    let tx = Tap::new(interp, sx, c[0].w, c[0].step);
                    let ty = Tap::new(interp, sy, c[0].h, 1);
                    p0[x] = sample_taps(interp, &c[0], tx, ty);
                    p1[x] = sample_taps(interp, &c[1], tx, ty);
                    p2[x] = sample_taps(interp, &c[2], tx, ty);
                }
            }
        }
        self.convert(&s.planes, &mut s.ch, 0, w);
        for (c, ch) in s.ch.iter_mut().enumerate().take(self.color.channels()) {
            let pad = self.pad[c];
            for (v, &o) in ch[..w].iter_mut().zip(&s.mask[..w]) {
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
                    *o = clamp255((y - k.y_off) * k.y_scale);
                }
            }
            (Src::Gray(_), ColorOrder::Gray) => c0.copy_from_slice(p0),
            (Src::Gray(_), _) => {
                c0.copy_from_slice(p0);
                c1.copy_from_slice(p0);
                c2.copy_from_slice(p0);
            }
            (Src::Rgb(_), ColorOrder::Rgb) => {
                c0.copy_from_slice(p0);
                c1.copy_from_slice(p1);
                c2.copy_from_slice(p2);
            }
            (Src::Rgb(_), ColorOrder::Bgr) => {
                c0.copy_from_slice(p2);
                c1.copy_from_slice(p1);
                c2.copy_from_slice(p0);
            }
            (Src::Rgb(_), ColorOrder::Gray) => {
                for (((o, &r), &g), &b) in c0.iter_mut().zip(p0).zip(p1).zip(p2) {
                    *o = 0.299 * r + 0.587 * g + 0.114 * b;
                }
            }
        }
    }
}

#[inline(always)]
fn clamp255(v: f32) -> f32 {
    v.clamp(0.0, 255.0)
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
        r[i] = clamp255(yp + k.rv * cr);
        g[i] = clamp255(yp - k.gu * cb - k.gv * cr);
        b[i] = clamp255(yp + k.bu * cb);
    }
}

/// Samples one channel at `(sx, sy)` (general path).
#[inline(always)]
fn sample2d(interp: Interp, c: &Chan<'_>, sx: f32, sy: f32) -> f32 {
    let tx = Tap::new(interp, sx, c.w, c.step);
    let ty = Tap::new(interp, sy, c.h, 1);
    sample_taps(interp, c, tx, ty)
}

/// Samples one channel with precomputed taps (`tx` in bytes, `ty` in rows).
#[inline(always)]
fn sample_taps(interp: Interp, c: &Chan<'_>, tx: Tap, ty: Tap) -> f32 {
    let r0 = c.row(ty.i0 as usize);
    match interp {
        Interp::Nearest => r0[tx.i0 as usize] as f32,
        Interp::Bilinear => {
            let r1 = c.row(ty.i1 as usize);
            let (a, b) = (tx.i0 as usize, tx.i1 as usize);
            let top = lerp(r0[a] as f32, r0[b] as f32, tx.f);
            let bot = lerp(r1[a] as f32, r1[b] as f32, tx.f);
            lerp(top, bot, ty.f)
        }
    }
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Horizontal gather + vertical blend of `N` channels that share the column
/// taps `t`: `out[k][i] = bilinear(r0[k], r1[k], t[i], fy)`.
/// For nearest, only `r0` and `i0` are used.
#[inline(always)]
fn gather<const N: usize>(
    interp: Interp,
    r0: [&[u8]; N],
    r1: [&[u8]; N],
    fy: f32,
    t: &ColTaps,
    out: [&mut [f32]; N],
) {
    let n = t.taps.len();
    for k in 0..N {
        // Establishes the invariant the unchecked loads below rely on.
        assert!(t.max < r0[k].len() && t.max < r1[k].len() && out[k].len() >= n);
    }
    let taps = &t.taps[..n];
    match interp {
        Interp::Nearest => {
            for (i, tap) in taps.iter().enumerate() {
                let a = tap.i0 as usize;
                for k in 0..N {
                    // SAFETY: `a <= t.max < r0[k].len()` (ColTaps invariant,
                    // asserted above).
                    let p = unsafe { *r0[k].get_unchecked(a) };
                    out[k][i] = p as f32;
                }
            }
        }
        Interp::Bilinear => {
            for (i, tap) in taps.iter().enumerate() {
                let (a, b, fx) = (tap.i0 as usize, tap.i1 as usize, tap.f);
                for k in 0..N {
                    // SAFETY: `a, b <= t.max`, which is smaller than both
                    // `r0[k].len()` and `r1[k].len()` (asserted above).
                    let (p00, p01, p10, p11) = unsafe {
                        (
                            *r0[k].get_unchecked(a),
                            *r0[k].get_unchecked(b),
                            *r1[k].get_unchecked(a),
                            *r1[k].get_unchecked(b),
                        )
                    };
                    let top = lerp(p00 as f32, p01 as f32, fx);
                    let bot = lerp(p10 as f32, p11 as f32, fx);
                    out[k][i] = lerp(top, bot, fy);
                }
            }
        }
    }
}
