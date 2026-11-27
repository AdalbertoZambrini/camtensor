//! Reference implementation and comparisons.

use camtensor_core::{
    Affine2, ColorMatrix, ColorOrder, ColorRange, ColorSpec, DType, Interp, KEYPOINT_TEMPLATE_112,
    Layout, Normalize, PixelFormat, PlaneDesc, ResizeMode, Roi, TensorSpec,
};
use camtensor_core::{Error, FrameDesc};
use half::f16;

use super::*;

// ---------------------------------------------------------------------------
// Reference: a direct, unoptimised transcription of the documented semantics.
// ---------------------------------------------------------------------------

/// Reads sample `(x, y)` of plane `p` with `step` bytes per sample and a
/// channel byte offset `co`.
fn px(data: &[u8], p: PlaneDesc, step: usize, co: usize, x: i64, y: i64) -> f32 {
    data[p.offset + y as usize * p.pitch + x as usize * step + co] as f32
}

#[allow(clippy::too_many_arguments)]
fn ref_sample(
    data: &[u8],
    p: PlaneDesc,
    step: usize,
    co: usize,
    w: usize,
    h: usize,
    sx: f32,
    sy: f32,
    interp: Interp,
) -> f32 {
    let (wm, hm) = (w as i64 - 1, h as i64 - 1);
    match interp {
        Interp::Nearest => {
            let xi = ((sx + 0.5).floor() as i64).clamp(0, wm);
            let yi = ((sy + 0.5).floor() as i64).clamp(0, hm);
            px(data, p, step, co, xi, yi)
        }
        Interp::Bilinear => {
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = sx - x0;
            let fy = sy - y0;
            let xa = (x0 as i64).clamp(0, wm);
            let xb = (x0 as i64 + 1).clamp(0, wm);
            let ya = (y0 as i64).clamp(0, hm);
            let yb = (y0 as i64 + 1).clamp(0, hm);
            (1.0 - fx) * (1.0 - fy) * px(data, p, step, co, xa, ya)
                + fx * (1.0 - fy) * px(data, p, step, co, xb, ya)
                + (1.0 - fx) * fy * px(data, p, step, co, xa, yb)
                + fx * fy * px(data, p, step, co, xb, yb)
        }
    }
}

/// Pixel value in pixel units, output channel order.
fn ref_pixel(desc: &FrameDesc, data: &[u8], spec: &TensorSpec, sx: f32, sy: f32) -> [f32; 3] {
    let (w, h) = (desc.width as usize, desc.height as usize);
    let pl = desc.planes;
    let interp = spec.interp;
    let (r, g, b, gray) = match desc.format {
        PixelFormat::Nv12 | PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
            let (shx, shy) = desc.format.chroma_shift();
            let cw = w.div_ceil(1 << shx);
            let ch = h.div_ceil(1 << shy);
            let yv = ref_sample(data, pl[0], 1, 0, w, h, sx, sy, interp);
            let cx = (sx + 0.5) / (1u32 << shx) as f32 - 0.5;
            let cy = (sy + 0.5) / (1u32 << shy) as f32 - 0.5;
            let (u, v) = if desc.format == PixelFormat::Nv12 {
                (
                    ref_sample(data, pl[1], 2, 0, cw, ch, cx, cy, interp),
                    ref_sample(data, pl[1], 2, 1, cw, ch, cx, cy, interp),
                )
            } else {
                (
                    ref_sample(data, pl[1], 1, 0, cw, ch, cx, cy, interp),
                    ref_sample(data, pl[2], 1, 0, cw, ch, cx, cy, interp),
                )
            };
            let (yp, cb, cr) = match desc.color.range {
                ColorRange::Limited => (
                    (yv - 16.0) * (255.0 / 219.0),
                    (u - 128.0) * (255.0 / 224.0),
                    (v - 128.0) * (255.0 / 224.0),
                ),
                ColorRange::Full => (yv, u - 128.0, v - 128.0),
            };
            let (r, g, b) = match desc.color.matrix {
                ColorMatrix::Bt601 => {
                    (yp + 1.402 * cr, yp - 0.344136 * cb - 0.714136 * cr, yp + 1.772 * cb)
                }
                ColorMatrix::Bt709 => {
                    (yp + 1.5748 * cr, yp - 0.187324 * cb - 0.468124 * cr, yp + 1.8556 * cb)
                }
            };
            (r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), b.clamp(0.0, 255.0), yp.clamp(0.0, 255.0))
        }
        PixelFormat::Gray8 => {
            let v = ref_sample(data, pl[0], 1, 0, w, h, sx, sy, interp);
            (v, v, v, v)
        }
        PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
            let s = |co| ref_sample(data, pl[0], 3, co, w, h, sx, sy, interp);
            let (r, g, b) = if desc.format == PixelFormat::Rgb24 {
                (s(0), s(1), s(2))
            } else {
                (s(2), s(1), s(0))
            };
            (r, g, b, 0.299 * r + 0.587 * g + 0.114 * b)
        }
    };
    match spec.color {
        ColorOrder::Rgb => [r, g, b],
        ColorOrder::Bgr => [b, g, r],
        ColorOrder::Gray => [gray, 0.0, 0.0],
    }
}

/// Reference output as element values in canonical `[c][y][x]` order.
fn reference(desc: &FrameDesc, data: &[u8], m: &Affine2, spec: &TensorSpec) -> Vec<f32> {
    let (ow, oh, c) = (spec.width as usize, spec.height as usize, spec.channels());
    let (sw, sh) = (desc.width as f32, desc.height as f32);
    let (na, nb) = spec.normalize.affine_coeffs();
    let [a, b, cc, d, e, f] = m.0;
    let mut out = vec![0.0; c * oh * ow];
    for y in 0..oh {
        for x in 0..ow {
            let (xf, yf) = (x as f32, y as f32);
            let sx = a * xf + b * yf + cc;
            let sy = d * xf + e * yf + f;
            let v = if sx < -0.5 || sx > sw - 0.5 || sy < -0.5 || sy > sh - 0.5 {
                spec.pad
            } else {
                ref_pixel(desc, data, spec, sx, sy)
            };
            for k in 0..c {
                out[k * oh * ow + y * ow + x] = match spec.dtype {
                    DType::F32 => v[k] * na[k] + nb[k],
                    DType::F16 => f16::from_f32(v[k] * na[k] + nb[k]).to_f32(),
                    DType::U8 => (v[k] + 0.5).floor().clamp(0.0, 255.0),
                };
            }
        }
    }
    out
}

/// Decodes an output buffer into canonical `[c][y][x]` order.
fn decode(out: &[u8], spec: &TensorSpec) -> Vec<f32> {
    let (w, h, c) = (spec.width as usize, spec.height as usize, spec.channels());
    let mut v = vec![0.0; c * h * w];
    for k in 0..c {
        for y in 0..h {
            for x in 0..w {
                let i = match spec.layout {
                    Layout::Nchw => k * h * w + y * w + x,
                    Layout::Nhwc => (y * w + x) * c + k,
                };
                v[k * h * w + y * w + x] = match spec.dtype {
                    DType::F32 => f32::from_ne_bytes(out[i * 4..i * 4 + 4].try_into().unwrap()),
                    DType::F16 => f16::from_bits(u16::from_ne_bytes(
                        out[i * 2..i * 2 + 2].try_into().unwrap(),
                    ))
                    .to_f32(),
                    DType::U8 => out[i] as f32,
                };
            }
        }
    }
    v
}

fn run(desc: &FrameDesc, data: &[u8], m: &Affine2, spec: &TensorSpec) -> Vec<u8> {
    let mut out = vec![0u8; spec.image_bytes()];
    preprocess_into(desc, data, m, spec, &mut out).expect("preprocess");
    out
}

/// Compares optimised output against the reference with per-dtype tolerances.
fn assert_matches(got: &[f32], want: &[f32], spec: &TensorSpec, ctx: &str) {
    let (w, h) = (spec.width as usize, spec.height as usize);
    let (na, _) = spec.normalize.affine_coeffs();
    for (i, (&g, &r)) in got.iter().zip(want).enumerate() {
        let c = i / (w * h);
        let tol = match spec.dtype {
            // 1e-3 pixel units, plus f32 rounding of the normalised value.
            DType::F32 => 1e-3 * na[c].abs() + 1e-6 * r.abs(),
            // Same, plus one f16 ulp (rounding may go either way).
            DType::F16 => 1e-3 * na[c].abs() + r.abs() / 1024.0 + 1e-7,
            DType::U8 => 1.0,
        };
        assert!(
            (g - r).abs() <= tol,
            "{ctx}: mismatch at c={c} y={} x={}: got {g}, want {r} (tol {tol})",
            (i / w) % h,
            i % w
        );
    }
}

// ---------------------------------------------------------------------------
// Random test inputs
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

const FORMATS: [PixelFormat; 7] = [
    PixelFormat::Nv12,
    PixelFormat::Yuv420p,
    PixelFormat::Yuv422p,
    PixelFormat::Yuv444p,
    PixelFormat::Gray8,
    PixelFormat::Rgb24,
    PixelFormat::Bgr24,
];

/// Builds a frame with random pitch padding and plane gaps, filled by `fill`.
fn frame_with(
    rng: &mut Rng,
    w: u32,
    h: u32,
    format: PixelFormat,
    color: ColorSpec,
    padded: bool,
    mut fill: impl FnMut(&mut Rng) -> u8,
) -> (FrameDesc, Vec<u8>) {
    let (wu, hu) = (w as usize, h as usize);
    let (sx, sy) = format.chroma_shift();
    let (cw, ch) = (wu.div_ceil(1 << sx), hu.div_ceil(1 << sy));
    let planes: Vec<(usize, usize)> = match format {
        PixelFormat::Nv12 => vec![(hu, wu), (ch, cw * 2)],
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
            vec![(hu, wu), (ch, cw), (ch, cw)]
        }
        PixelFormat::Gray8 => vec![(hu, wu)],
        PixelFormat::Rgb24 | PixelFormat::Bgr24 => vec![(hu, wu * 3)],
    };
    let mut desc = FrameDesc::packed(w, h, format, color);
    let mut off = 0;
    for (i, &(rows, row_bytes)) in planes.iter().enumerate() {
        let (gap, extra) = if padded { (rng.below(8), rng.below(10)) } else { (0, 0) };
        off += gap;
        desc.planes[i] = PlaneDesc { offset: off, pitch: row_bytes + extra };
        off += (row_bytes + extra) * rows;
    }
    let data = (0..off).map(|_| fill(rng)).collect();
    (desc, data)
}

fn random_frame(rng: &mut Rng, w: u32, h: u32, format: PixelFormat) -> (FrameDesc, Vec<u8>) {
    let color = ColorSpec {
        range: rng.pick(&[ColorRange::Limited, ColorRange::Full]),
        matrix: rng.pick(&[ColorMatrix::Bt601, ColorMatrix::Bt709]),
    };
    frame_with(rng, w, h, format, color, true, |r| r.next() as u8)
}

fn random_spec(rng: &mut Rng, w: u32, h: u32) -> TensorSpec {
    TensorSpec {
        width: w,
        height: h,
        dtype: rng.pick(&[DType::F32, DType::F16, DType::U8]),
        layout: rng.pick(&[Layout::Nchw, Layout::Nhwc]),
        color: rng.pick(&[ColorOrder::Rgb, ColorOrder::Bgr, ColorOrder::Gray]),
        resize: rng.pick(&[
            ResizeMode::Stretch,
            ResizeMode::Letterbox,
            ResizeMode::LetterboxTopLeft,
            ResizeMode::CropCenter,
            ResizeMode::None,
        ]),
        interp: rng.pick(&[Interp::Bilinear, Interp::Nearest]),
        normalize: rng.pick(&[
            Normalize::NONE,
            Normalize::UNIT,
            Normalize::IMAGENET,
            Normalize::SYMMETRIC,
        ]),
        pad: [rng.uniform(0.0, 255.0), rng.uniform(0.0, 255.0), rng.uniform(0.0, 255.0)],
    }
}

/// Random sampling matrix of one of several families.
fn random_matrix(rng: &mut Rng, src: (u32, u32), spec: &TensorSpec) -> (Affine2, &'static str) {
    let (sw, sh) = (src.0 as f32, src.1 as f32);
    match rng.below(5) {
        0 => (plan(src.0, src.1, spec).sampling, "plan"),
        1 => {
            // Axis-aligned, possibly flipped, partially covering.
            let a = rng.uniform(0.1, 4.0) * if rng.below(4) == 0 { -1.0 } else { 1.0 };
            let e = rng.uniform(0.1, 4.0) * if rng.below(4) == 0 { -1.0 } else { 1.0 };
            let c = rng.uniform(-0.5 * sw, sw);
            let f = rng.uniform(-0.5 * sh, sh);
            (Affine2([a, 0.0, c, 0.0, e, f]), "axis")
        }
        2 => {
            // Rotation + scale + translation.
            let t = rng.uniform(-3.2, 3.2);
            let s = rng.uniform(0.2, 3.0);
            let (c, si) = (t.cos() * s, t.sin() * s);
            let tx = rng.uniform(-0.3 * sw, 1.2 * sw);
            let ty = rng.uniform(-0.3 * sh, 1.2 * sh);
            (Affine2([c, -si, tx, si, c, ty]), "rotation")
        }
        3 => {
            // Nearly axis-aligned (exercises the general path on resizes).
            let mut m = plan(src.0, src.1, spec).sampling;
            m.0[1] = rng.uniform(-1e-3, 1e-3);
            m.0[3] = rng.uniform(-1e-3, 1e-3);
            (m, "skewed-plan")
        }
        _ => {
            // Random box crop through plan_roi.
            let x = rng.uniform(-0.2 * sw, 0.8 * sw);
            let y = rng.uniform(-0.2 * sh, 0.8 * sh);
            let roi = Roi::Box { x, y, w: rng.uniform(1.0, sw), h: rng.uniform(1.0, sh) };
            (plan_roi(&roi, spec).unwrap().sampling, "box")
        }
    }
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

#[test]
fn matches_reference_random() {
    let mut rng = Rng(0x00C0_FFEE);
    for case in 0..1500 {
        let format = FORMATS[case % FORMATS.len()];
        let (sw, sh) = (rng.range(1, 70) as u32, rng.range(1, 70) as u32);
        let (ow, oh) = (rng.range(1, 48) as u32, rng.range(1, 48) as u32);
        let (desc, data) = random_frame(&mut rng, sw, sh, format);
        let spec = random_spec(&mut rng, ow, oh);
        let (m, kind) = random_matrix(&mut rng, (sw, sh), &spec);
        let ctx = format!("case {case}: {format:?} {sw}x{sh} -> {ow}x{oh} {kind} {m:?} {spec:?}");
        let got = decode(&run(&desc, &data, &m, &spec), &spec);
        let want = reference(&desc, &data, &m, &spec);
        assert_matches(&got, &want, &spec, &ctx);
    }
}

#[test]
fn matches_reference_odd_sizes() {
    let mut rng = Rng(7);
    for (i, &format) in FORMATS.iter().enumerate() {
        let (desc, data) = random_frame(&mut rng, 641, 359, format);
        let mut spec = random_spec(&mut rng, 97, 61);
        spec.interp = if i % 2 == 0 { Interp::Bilinear } else { Interp::Nearest };
        for resize in [ResizeMode::Letterbox, ResizeMode::CropCenter, ResizeMode::Stretch] {
            spec.resize = resize;
            let m = plan(641, 359, &spec).sampling;
            let got = decode(&run(&desc, &data, &m, &spec), &spec);
            let want = reference(&desc, &data, &m, &spec);
            assert_matches(&got, &want, &spec, &format!("{format:?} 641x359 {spec:?}"));
        }
        // Upscale of the odd-sized frame, plus a rotation.
        let mut spec = random_spec(&mut rng, 701, 13);
        spec.resize = ResizeMode::Stretch;
        let m = plan(641, 359, &spec).sampling;
        let got = decode(&run(&desc, &data, &m, &spec), &spec);
        assert_matches(&got, &reference(&desc, &data, &m, &spec), &spec, "upscale");
        let (m, _) = (Affine2([0.8, -0.6, 300.0, 0.6, 0.8, -20.0]), "rot");
        let got = decode(&run(&desc, &data, &m, &spec), &spec);
        assert_matches(&got, &reference(&desc, &data, &m, &spec), &spec, "rotation");
    }
}

/// Strong downscales take the direct 4-tap gather instead of row blending.
#[test]
fn matches_reference_strong_downscale() {
    let mut rng = Rng(42);
    for &format in &FORMATS {
        let (desc, data) = random_frame(&mut rng, 613, 401, format);
        for (ow, oh) in [(37, 23), (101, 67), (613, 401)] {
            let mut spec = random_spec(&mut rng, ow, oh);
            spec.interp = Interp::Bilinear;
            spec.resize = ResizeMode::Stretch;
            let m = plan(613, 401, &spec).sampling;
            let got = decode(&run(&desc, &data, &m, &spec), &spec);
            let want = reference(&desc, &data, &m, &spec);
            assert_matches(&got, &want, &spec, &format!("{format:?} -> {ow}x{oh} {spec:?}"));
        }
    }
}

#[test]
fn rayon_matches_single_threaded() {
    let mut rng = Rng(99);
    for &format in &FORMATS {
        let (desc, data) = random_frame(&mut rng, 333, 217, format);
        let spec = random_spec(&mut rng, 301, 257);
        let (m, _) = random_matrix(&mut rng, (333, 217), &spec);
        let single = run(&desc, &data, &m, &spec);
        let mut par = vec![0u8; spec.image_bytes()];
        preprocess_into_with(&desc, &data, &m, &spec, &mut par, Parallelism::Rayon).unwrap();
        assert!(single == par, "{format:?} {spec:?}");
    }
}

// ---------------------------------------------------------------------------
// Known values
// ---------------------------------------------------------------------------

fn raw_spec(w: u32, h: u32) -> TensorSpec {
    TensorSpec { width: w, height: h, normalize: Normalize::NONE, ..TensorSpec::default() }
}

fn solid_yuv(format: PixelFormat, color: ColorSpec, yuv: [u8; 3]) -> (FrameDesc, Vec<u8>) {
    let (w, h) = (6u32, 4u32);
    let desc = FrameDesc::packed(w, h, format, color);
    let mut data = vec![0u8; desc.byte_len()];
    let luma = w as usize * h as usize;
    data[..luma].fill(yuv[0]);
    match format {
        PixelFormat::Nv12 => {
            for uv in data[luma..].as_chunks_mut::<2>().0 {
                uv[0] = yuv[1];
                uv[1] = yuv[2];
            }
        }
        _ => {
            let c = (desc.byte_len() - luma) / 2;
            data[luma..luma + c].fill(yuv[1]);
            data[luma + c..].fill(yuv[2]);
        }
    }
    (desc, data)
}

#[test]
fn solid_yuv_colours() {
    use ColorMatrix::*;
    use ColorRange::*;
    // (range, matrix, yuv, expected rgb (rounded), tolerance)
    type Case = (ColorRange, ColorMatrix, [u8; 3], [f32; 3], f32);
    let cases: &[Case] = &[
        (Limited, Bt601, [16, 128, 128], [0.0, 0.0, 0.0], 1e-3),
        (Limited, Bt601, [235, 128, 128], [255.0, 255.0, 255.0], 1e-3),
        (Limited, Bt709, [16, 128, 128], [0.0, 0.0, 0.0], 1e-3),
        (Limited, Bt709, [235, 128, 128], [255.0, 255.0, 255.0], 1e-3),
        (Full, Bt601, [0, 128, 128], [0.0, 0.0, 0.0], 1e-3),
        (Full, Bt601, [255, 128, 128], [255.0, 255.0, 255.0], 1e-3),
        (Full, Bt709, [100, 128, 128], [100.0, 100.0, 100.0], 1e-3),
        // Primaries (published 8-bit encodings), within quantisation error.
        (Limited, Bt601, [81, 90, 240], [255.0, 0.0, 0.0], 2.0),
        (Limited, Bt601, [145, 54, 34], [0.0, 255.0, 0.0], 2.0),
        (Limited, Bt601, [41, 240, 110], [0.0, 0.0, 255.0], 2.0),
        (Limited, Bt709, [63, 102, 240], [255.0, 0.0, 0.0], 2.0),
        (Limited, Bt709, [173, 42, 26], [0.0, 255.0, 0.0], 2.0),
        (Limited, Bt709, [32, 240, 118], [0.0, 0.0, 255.0], 2.0),
        (Full, Bt601, [76, 85, 255], [255.0, 0.0, 0.0], 2.0),
        (Full, Bt709, [54, 99, 255], [255.0, 0.0, 0.0], 2.0),
        // Exact formula values: limited BT.601, Y=126 U=100 V=180.
        (
            Limited,
            Bt601,
            [126, 100, 180],
            {
                let yp = 110.0 * 255.0 / 219.0;
                let (cb, cr) = (-28.0 * 255.0 / 224.0, 52.0 * 255.0 / 224.0);
                [yp + 1.402 * cr, yp - 0.344136 * cb - 0.714136 * cr, yp + 1.772 * cb]
            },
            1e-3,
        ),
        // Exact formula values: full BT.709, Y=90 U=160 V=70.
        (
            Full,
            Bt709,
            [90, 160, 70],
            {
                let (yp, cb, cr) = (90.0f32, 32.0f32, -58.0f32);
                [yp + 1.5748 * cr, yp - 0.187324 * cb - 0.468124 * cr, yp + 1.8556 * cb]
            },
            1e-3,
        ),
    ];
    for &(range, matrix, yuv, rgb, tol) in cases {
        for format in
            [PixelFormat::Nv12, PixelFormat::Yuv420p, PixelFormat::Yuv422p, PixelFormat::Yuv444p]
        {
            let (desc, data) = solid_yuv(format, ColorSpec { range, matrix }, yuv);
            for (color, order) in [(ColorOrder::Rgb, [0, 1, 2]), (ColorOrder::Bgr, [2, 1, 0])] {
                let spec = TensorSpec { color, resize: ResizeMode::Stretch, ..raw_spec(5, 3) };
                let m = plan(desc.width, desc.height, &spec).sampling;
                let got = decode(&run(&desc, &data, &m, &spec), &spec);
                for (k, &o) in order.iter().enumerate() {
                    let want = rgb[o].clamp(0.0, 255.0);
                    for &v in &got[k * 15..(k + 1) * 15] {
                        assert!(
                            (v - want).abs() <= tol,
                            "{range:?} {matrix:?} {yuv:?} {format:?} {color:?} ch{k}: {v} vs {want}"
                        );
                    }
                }
            }
            // Gray output is clamp(Y').
            let spec = TensorSpec { color: ColorOrder::Gray, ..raw_spec(5, 3) };
            let m = plan(desc.width, desc.height, &spec).sampling;
            let got = decode(&run(&desc, &data, &m, &spec), &spec);
            let yp = match range {
                Limited => (yuv[0] as f32 - 16.0) * (255.0 / 219.0),
                Full => yuv[0] as f32,
            };
            assert!(got.iter().all(|&v| (v - yp.clamp(0.0, 255.0)).abs() < 1e-3), "{got:?}");
        }
    }
}

#[test]
fn letterbox_1080p_pad_rows() {
    let desc = FrameDesc::packed(1920, 1080, PixelFormat::Nv12, ColorSpec::default());
    let mut data = vec![128u8; desc.byte_len()];
    data[..1920 * 1080].fill(200);
    for dtype in [DType::F32, DType::F16] {
        for layout in [Layout::Nchw, Layout::Nhwc] {
            let spec = TensorSpec { dtype, layout, ..TensorSpec::default() };
            let p = plan(1920, 1080, &spec);
            let got = decode(&run(&desc, &data, &p.sampling, &spec), &spec);
            let (a, b) = spec.normalize.affine_coeffs();
            for c in 0..3 {
                let pad = match dtype {
                    DType::F16 => f16::from_f32(114.0 * a[c] + b[c]).to_f32(),
                    _ => 114.0 * a[c] + b[c],
                };
                let img = &got[c * 640 * 640..(c + 1) * 640 * 640];
                for y in (0..140).chain(500..640) {
                    assert!(img[y * 640..(y + 1) * 640].iter().all(|&v| v == pad), "row {y}");
                }
                for y in 140..500 {
                    assert!(img[y * 640..(y + 1) * 640].iter().all(|&v| v != pad), "row {y}");
                }
            }
        }
    }
}

/// RGB24 frame whose pixel `(x, y)` encodes its own coordinates.
fn coord_frame(w: u32, h: u32) -> (FrameDesc, Vec<u8>) {
    let desc = FrameDesc::packed(w, h, PixelFormat::Rgb24, ColorSpec::default());
    let mut data = vec![0u8; desc.byte_len()];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = &mut data[(y * w as usize + x) * 3..][..3];
            p.copy_from_slice(&[(x % 251) as u8, (y % 241) as u8, ((x * 7 + y * 13) % 256) as u8]);
        }
    }
    (desc, data)
}

#[test]
fn crop_center_has_no_padding() {
    let desc = FrameDesc::packed(1920, 1080, PixelFormat::Yuv420p, ColorSpec::default());
    let data = vec![50u8; desc.byte_len()];
    let spec = TensorSpec {
        resize: ResizeMode::CropCenter,
        dtype: DType::U8,
        pad: [255.0; 3],
        ..TensorSpec::default()
    };
    let p = plan(1920, 1080, &spec);
    let (cx, cy) = p.dst_from_src.apply(960.0, 540.0);
    assert!((cx - 320.0).abs() < 1e-3 && (cy - 320.0).abs() < 1e-3);
    let out = run(&desc, &data, &p.sampling, &spec);
    assert!(out.iter().all(|&v| v != 255));
    // A coordinate-coded frame: the centre output column samples x ~ 960.
    let (desc, data) = coord_frame(1920, 1080);
    let spec = TensorSpec { dtype: DType::F32, ..spec };
    let got = decode(&run(&desc, &data, &p.sampling, &spec), &spec);
    let want = reference(&desc, &data, &p.sampling, &spec);
    assert_matches(&got, &want, &spec, "crop center");
}

#[test]
fn stretch_halves_blocks_exactly() {
    // 2x2 constant blocks: downscaling by 2 samples block centres exactly.
    let (w, h) = (64u32, 32u32);
    let desc = FrameDesc::packed(w, h, PixelFormat::Rgb24, ColorSpec::default());
    let mut data = vec![0u8; desc.byte_len()];
    let block = |bx: usize, by: usize, c: usize| ((bx * 31 + by * 17 + c * 5) % 256) as u8;
    for y in 0..h as usize {
        for x in 0..w as usize {
            for c in 0..3 {
                data[(y * w as usize + x) * 3 + c] = block(x / 2, y / 2, c);
            }
        }
    }
    let spec = TensorSpec { resize: ResizeMode::Stretch, dtype: DType::U8, ..raw_spec(32, 16) };
    let out = run(&desc, &data, &plan(w, h, &spec).sampling, &spec);
    for c in 0..3 {
        for y in 0..16 {
            for x in 0..32 {
                assert_eq!(out[c * 512 + y * 32 + x], block(x, y, c), "c{c} y{y} x{x}");
            }
        }
    }
}

#[test]
fn resize_none_copies_top_left() {
    let mut rng = Rng(3);
    let (desc, data) =
        frame_with(&mut rng, 50, 40, PixelFormat::Gray8, ColorSpec::default(), true, |r| {
            r.next() as u8
        });
    for interp in [Interp::Bilinear, Interp::Nearest] {
        let spec = TensorSpec {
            resize: ResizeMode::None,
            dtype: DType::U8,
            color: ColorOrder::Gray,
            interp,
            pad: [7.0; 3],
            ..raw_spec(64, 32)
        };
        let out = run(&desc, &data, &plan(50, 40, &spec).sampling, &spec);
        let p = desc.planes[0];
        for y in 0..32 {
            for x in 0..64 {
                let want = if x < 50 { data[p.offset + y * p.pitch + x] } else { 7 };
                assert_eq!(out[y * 64 + x], want, "y{y} x{x}");
            }
        }
    }
}

#[test]
fn nhwc_equals_transposed_nchw() {
    let mut rng = Rng(11);
    for &format in &FORMATS {
        let (desc, data) = random_frame(&mut rng, 37, 23, format);
        for dtype in [DType::F32, DType::F16, DType::U8] {
            for color in [ColorOrder::Rgb, ColorOrder::Gray] {
                let nchw = TensorSpec {
                    dtype,
                    color,
                    layout: Layout::Nchw,
                    normalize: Normalize::IMAGENET,
                    ..TensorSpec::default()
                };
                let nchw = TensorSpec { width: 29, height: 31, ..nchw };
                let nhwc = TensorSpec { layout: Layout::Nhwc, ..nchw };
                let m = plan(37, 23, &nchw).sampling;
                let (a, b) = (run(&desc, &data, &m, &nchw), run(&desc, &data, &m, &nhwc));
                let (c, hw, es) = (nchw.channels(), 29 * 31, dtype.size());
                for k in 0..c {
                    for i in 0..hw {
                        let ia = (k * hw + i) * es;
                        let ib = (i * c + k) * es;
                        assert_eq!(a[ia..ia + es], b[ib..ib + es], "{format:?} {dtype:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn gray_output_and_bgr_swap() {
    let (desc, data) = coord_frame(16, 8);
    let id = |spec: &TensorSpec| plan(16, 8, spec).sampling;
    let rgb = TensorSpec { resize: ResizeMode::None, ..raw_spec(16, 8) };
    let bgr = TensorSpec { color: ColorOrder::Bgr, ..rgb };
    let gray = TensorSpec { color: ColorOrder::Gray, ..rgb };
    let r = decode(&run(&desc, &data, &id(&rgb), &rgb), &rgb);
    let b = decode(&run(&desc, &data, &id(&bgr), &bgr), &bgr);
    let g = decode(&run(&desc, &data, &id(&gray), &gray), &gray);
    let n = 16 * 8;
    for i in 0..n {
        let px = &data[i * 3..i * 3 + 3];
        assert_eq!(r[i], px[0] as f32);
        assert_eq!(r[n + i], px[1] as f32);
        assert_eq!(r[2 * n + i], px[2] as f32);
        assert_eq!((b[i], b[n + i], b[2 * n + i]), (r[2 * n + i], r[n + i], r[i]));
        let want = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
        assert!((g[i] - want).abs() < 1e-3);
    }
    // A BGR24 source with BGR output is the identity on bytes.
    let mut bdesc = desc;
    bdesc.format = PixelFormat::Bgr24;
    let u8bgr = TensorSpec { dtype: DType::U8, layout: Layout::Nhwc, ..bgr };
    assert_eq!(run(&bdesc, &data, &id(&u8bgr), &u8bgr), data);
    // Gray8 source is replicated.
    let gdesc = FrameDesc::packed(16, 8, PixelFormat::Gray8, ColorSpec::default());
    let gdata: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
    let out = decode(&run(&gdesc, &gdata, &id(&rgb), &rgb), &rgb);
    for i in 0..n {
        let v = gdata[i] as f32;
        assert_eq!((out[i], out[n + i], out[2 * n + i]), (v, v, v));
    }
}

#[test]
fn template_keypoints_give_identity_crop() {
    let mut rng = Rng(5);
    let (desc, data) =
        frame_with(&mut rng, 112, 112, PixelFormat::Rgb24, ColorSpec::default(), false, |r| {
            r.next() as u8
        });
    let spec = TensorSpec {
        normalize: Normalize::NONE,
        ..TensorSpec { width: 112, height: 112, ..TensorSpec::default() }
    };
    let p = plan_roi(&Roi::Keypoints { points: KEYPOINT_TEMPLATE_112 }, &spec).unwrap();
    let got = decode(&run(&desc, &data, &p.sampling, &spec), &spec);
    let n = 112 * 112;
    for c in 0..3 {
        for i in 0..n {
            let want = data[i * 3 + c] as f32;
            assert!(
                (got[c * n + i] - want).abs() < 0.05,
                "c{c} i{i}: {} vs {want}",
                got[c * n + i]
            );
        }
    }
    // Same through the u8 path (rounding absorbs the tiny matrix error).
    let u8spec = TensorSpec { dtype: DType::U8, layout: Layout::Nhwc, ..spec };
    assert_eq!(run(&desc, &data, &p.sampling, &u8spec), data);
}

#[test]
fn rejects_invalid_inputs() {
    let desc = FrameDesc::packed(64, 48, PixelFormat::Nv12, ColorSpec::default());
    let data = vec![0u8; desc.byte_len()];
    let spec = raw_spec(32, 32);
    let m = plan(64, 48, &spec).sampling;
    let mut out = vec![0u8; spec.image_bytes()];
    let is_pre = |r: Result<()>| matches!(r, Err(Error::Preprocess(_)));

    assert!(is_pre(preprocess_into(&desc, &data, &m, &spec, &mut out[1..])));
    assert!(is_pre(preprocess_into(&desc, &data[..data.len() - 1], &m, &spec, &mut out)));
    let mut bad = desc;
    bad.planes[1].pitch = 10;
    assert!(is_pre(preprocess_into(&bad, &data, &m, &spec, &mut out)));
    let mut bad = desc;
    bad.planes[1].offset = usize::MAX - 4;
    assert!(is_pre(preprocess_into(&bad, &data, &m, &spec, &mut out)));
    let mut bad = desc;
    bad.width = 0;
    assert!(is_pre(preprocess_into(&bad, &data, &m, &spec, &mut out)));
    let nan = Affine2([f32::NAN, 0.0, 0.0, 0.0, 1.0, 0.0]);
    assert!(is_pre(preprocess_into(&desc, &data, &nan, &spec, &mut out)));
    let zero = TensorSpec { width: 0, ..spec };
    assert!(is_pre(preprocess_into(&desc, &data, &m, &zero, &mut [])));
    assert!(plan_roi(&Roi::Box { x: 0.0, y: 0.0, w: 0.0, h: 5.0 }, &spec).is_err());
    // Huge but finite matrices must not panic.
    let huge = Affine2([3e38, 3e38, -3e38, 1e30, -1e30, 0.0]);
    preprocess_into(&desc, &data, &huge, &spec, &mut out).unwrap();
}

#[test]
fn tiny_sources_and_outputs() {
    let mut rng = Rng(1);
    for &format in &FORMATS {
        for (sw, sh) in [(1, 1), (1, 5), (5, 1), (2, 3)] {
            let (desc, data) = random_frame(&mut rng, sw, sh, format);
            for (ow, oh) in [(1, 1), (3, 1), (1, 4), (9, 7)] {
                let spec = random_spec(&mut rng, ow, oh);
                for m in [plan(sw, sh, &spec).sampling, random_matrix(&mut rng, (sw, sh), &spec).0]
                {
                    let got = decode(&run(&desc, &data, &m, &spec), &spec);
                    let want = reference(&desc, &data, &m, &spec);
                    assert_matches(&got, &want, &spec, &format!("{format:?} {sw}x{sh} {spec:?}"));
                }
            }
        }
    }
}
