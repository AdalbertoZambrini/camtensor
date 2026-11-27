//! GPU tests of the fused preprocessing kernel against a straightforward CPU
//! reference of the documented semantics. Skipped when no CUDA device is
//! present.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use std::sync::Arc;

use camtensor_core::{
    Affine2, ColorMatrix, ColorOrder, ColorRange, ColorSpec, DType, FrameDesc, Interp, Layout,
    Normalize, PixelFormat, ResizeMode, TensorSpec,
};
use camtensor_cuda::{CudaDevice, CudaStream, DeviceBuffer, Preprocessor, SourceImage};

struct Gpu {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    pre: Arc<Preprocessor>,
}

fn gpu() -> Option<Gpu> {
    if !camtensor_cuda::is_available() {
        eprintln!("skipping: no CUDA device");
        return None;
    }
    let device = CudaDevice::get(0).expect("device");
    let stream = CudaStream::new(&device).expect("stream");
    let pre = Preprocessor::get(&device).expect("kernel load");
    Some(Gpu { device, stream, pre })
}

/// Deterministic pseudo-random bytes with some spatial structure.
fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..len)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let smooth = ((i as u32 / 7) ^ (i as u32 / 1931)) as u8;
            smooth.wrapping_add((s >> 27) as u8)
        })
        .collect()
}

// ---------------------------------------------------------------- reference

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn sample_plane(
    data: &[u8],
    offset: usize,
    pitch: usize,
    w: usize,
    h: usize,
    step: usize,
    ch: usize,
    sx: f32,
    sy: f32,
    interp: Interp,
) -> f32 {
    let at = |x: i64, y: i64| -> f32 {
        let x = x.clamp(0, w as i64 - 1) as usize;
        let y = y.clamp(0, h as i64 - 1) as usize;
        data[offset + y * pitch + x * step + ch] as f32
    };
    match interp {
        Interp::Nearest => at((sx + 0.5).floor() as i64, (sy + 0.5).floor() as i64),
        Interp::Bilinear => {
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let (gx, gy) = (1.0 - fx, 1.0 - fy);
            gx * gy * at(x0, y0)
                + fx * gy * at(x0 + 1, y0)
                + gx * fy * at(x0, y0 + 1)
                + fx * fy * at(x0 + 1, y0 + 1)
        }
    }
}

/// CPU reference producing output values in pixel units (before
/// normalisation), output channel order, `[c][y][x]`.
fn reference_pixels(src: &[u8], desc: &FrameDesc, m: &Affine2, spec: &TensorSpec) -> Vec<Vec<f32>> {
    let (w, h) = (desc.width as usize, desc.height as usize);
    let (shx, shy) = desc.format.chroma_shift();
    let (cw, chh) = ((w + (1 << shx) - 1) >> shx, (h + (1 << shy) - 1) >> shy);
    let nc = spec.channels();
    let mut out = vec![vec![0f32; (spec.width * spec.height) as usize]; nc];
    let (cr_r, cb_g, cr_g, cb_b) = match desc.color.matrix {
        ColorMatrix::Bt601 => (1.402f32, 0.344136f32, 0.714136f32, 1.772f32),
        ColorMatrix::Bt709 => (1.5748, 0.187324, 0.468124, 1.8556),
    };
    for y in 0..spec.height as usize {
        for x in 0..spec.width as usize {
            let (sx, sy) = m.apply(x as f32, y as f32);
            let outside = sx < -0.5 || sx > w as f32 - 0.5 || sy < -0.5 || sy > h as f32 - 0.5;
            let v = if outside {
                spec.pad
            } else {
                let p = &desc.planes;
                let (rgb, gray): ([f32; 3], Option<f32>) = if desc.format.is_yuv() {
                    let yy =
                        sample_plane(src, p[0].offset, p[0].pitch, w, h, 1, 0, sx, sy, spec.interp);
                    let (yp, cs) = match desc.color.range {
                        ColorRange::Limited => ((yy - 16.0) * (255.0 / 219.0), 255.0f32 / 224.0),
                        ColorRange::Full => (yy, 1.0),
                    };
                    let cx = (sx + 0.5) / (1 << shx) as f32 - 0.5;
                    let cy = (sy + 0.5) / (1 << shy) as f32 - 0.5;
                    let (u, vv) = if desc.format == PixelFormat::Nv12 {
                        (
                            sample_plane(
                                src,
                                p[1].offset,
                                p[1].pitch,
                                cw,
                                chh,
                                2,
                                0,
                                cx,
                                cy,
                                spec.interp,
                            ),
                            sample_plane(
                                src,
                                p[1].offset,
                                p[1].pitch,
                                cw,
                                chh,
                                2,
                                1,
                                cx,
                                cy,
                                spec.interp,
                            ),
                        )
                    } else {
                        (
                            sample_plane(
                                src,
                                p[1].offset,
                                p[1].pitch,
                                cw,
                                chh,
                                1,
                                0,
                                cx,
                                cy,
                                spec.interp,
                            ),
                            sample_plane(
                                src,
                                p[2].offset,
                                p[2].pitch,
                                cw,
                                chh,
                                1,
                                0,
                                cx,
                                cy,
                                spec.interp,
                            ),
                        )
                    };
                    let cb = (u - 128.0) * cs;
                    let cr = (vv - 128.0) * cs;
                    (
                        [
                            clampf(yp + cr_r * cr, 0.0, 255.0),
                            clampf(yp - cb_g * cb - cr_g * cr, 0.0, 255.0),
                            clampf(yp + cb_b * cb, 0.0, 255.0),
                        ],
                        Some(clampf(yp, 0.0, 255.0)),
                    )
                } else if desc.format == PixelFormat::Gray8 {
                    let g =
                        sample_plane(src, p[0].offset, p[0].pitch, w, h, 1, 0, sx, sy, spec.interp);
                    ([g, g, g], Some(g))
                } else {
                    let c: Vec<f32> = (0..3)
                        .map(|k| {
                            sample_plane(
                                src,
                                p[0].offset,
                                p[0].pitch,
                                w,
                                h,
                                3,
                                k,
                                sx,
                                sy,
                                spec.interp,
                            )
                        })
                        .collect();
                    let rgb = if desc.format == PixelFormat::Rgb24 {
                        [c[0], c[1], c[2]]
                    } else {
                        [c[2], c[1], c[0]]
                    };
                    (rgb, None)
                };
                match spec.color {
                    ColorOrder::Rgb => rgb,
                    ColorOrder::Bgr => [rgb[2], rgb[1], rgb[0]],
                    ColorOrder::Gray => {
                        [gray.unwrap_or(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]), 0.0, 0.0]
                    }
                }
            };
            for c in 0..nc {
                out[c][y * spec.width as usize + x] = v[c];
            }
        }
    }
    out
}

/// Expected output element (as f32) at `[c][y*W+x]`.
fn expected_value(px: f32, c: usize, spec: &TensorSpec) -> f32 {
    match spec.dtype {
        DType::U8 => clampf((px + 0.5).floor(), 0.0, 255.0),
        _ => {
            let (a, b) = spec.normalize.affine_coeffs();
            px * a[c] + b[c]
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        e => sign * (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Downloads a tensor image and returns it as `[c][y*W+x]` f32 values.
fn read_output(buf: &DeviceBuffer, offset: usize, spec: &TensorSpec) -> Vec<Vec<f32>> {
    let mut bytes = vec![0u8; spec.image_bytes()];
    buf.copy_to_host(offset, &mut bytes).unwrap();
    let (w, h, nc) = (spec.width as usize, spec.height as usize, spec.channels());
    let elem = |i: usize| -> f32 {
        match spec.dtype {
            DType::U8 => bytes[i] as f32,
            DType::F16 => f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])),
            DType::F32 => f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()),
        }
    };
    let mut out = vec![vec![0f32; w * h]; nc];
    for c in 0..nc {
        for p in 0..w * h {
            let i = match spec.layout {
                Layout::Nchw => c * w * h + p,
                Layout::Nhwc => p * nc + c,
            };
            out[c][p] = elem(i);
        }
    }
    out
}

struct Run {
    out: Vec<Vec<f32>>,
    reference: Vec<Vec<f32>>,
}

fn run(g: &Gpu, src: &[u8], desc: FrameDesc, dst_from_src: Affine2, spec: &TensorSpec) -> Run {
    let buf = DeviceBuffer::from_host(&g.device, src).unwrap();
    let img = SourceImage::from_buffer(&buf, desc).unwrap();
    let m = Affine2::sampling_matrix(&dst_from_src).unwrap();
    let out = DeviceBuffer::zeroed(&g.device, spec.image_bytes()).unwrap();
    g.pre.preprocess_into(&g.stream, &img, &[m], &out, 0, spec).unwrap();
    g.stream.synchronize().unwrap();
    let px = reference_pixels(src, &desc, &m, spec);
    let reference = px
        .iter()
        .enumerate()
        .map(|(c, ch)| ch.iter().map(|&v| expected_value(v, c, spec)).collect())
        .collect();
    Run { out: read_output(&out, 0, spec), reference }
}

/// Max abs difference and number of elements above `tol`.
fn compare(r: &Run, tol: f32) -> (f32, usize) {
    let mut max = 0f32;
    let mut bad = 0;
    for (a, b) in r.out.iter().flatten().zip(r.reference.iter().flatten()) {
        let d = (a - b).abs();
        max = max.max(d);
        if d > tol {
            bad += 1;
        }
    }
    (max, bad)
}

fn nv12_1080p(color: ColorSpec) -> (Vec<u8>, FrameDesc) {
    let desc = FrameDesc::packed(1920, 1080, PixelFormat::Nv12, color);
    (pattern(desc.byte_len(), 1), desc)
}

fn letterbox_spec() -> TensorSpec {
    TensorSpec { width: 640, height: 640, ..Default::default() }
}

fn letterbox_m(desc: &FrameDesc, spec: &TensorSpec) -> Affine2 {
    Affine2::fit(
        desc.width as f32,
        desc.height as f32,
        spec.width as f32,
        spec.height as f32,
        spec.resize,
    )
}

// -------------------------------------------------------------------- tests

#[test]
fn letterbox_1080p_nv12_f32_matches_reference_and_pads() {
    let Some(g) = gpu() else { return };
    let (src, desc) = nv12_1080p(ColorSpec::default());
    let spec = letterbox_spec();
    let r = run(&g, &src, desc, letterbox_m(&desc, &spec), &spec);
    let (max, bad) = compare(&r, 1e-4);
    assert_eq!(bad, 0, "max diff {max}");
    // Rows 0..140 and 500..640 are padding (scale 1/3, 360 image rows).
    let (a, b) = spec.normalize.affine_coeffs();
    let pad = 114.0 * a[0] + b[0];
    for c in 0..3 {
        for y in (0..140).chain(500..640) {
            for x in 0..640 {
                assert_eq!(r.out[c][y * 640 + x], pad, "pad at c{c} ({x},{y})");
            }
        }
        // The image rows are not padding.
        assert!((140..500).any(|y| r.out[c][y * 640 + 320] != pad));
    }
}

#[test]
fn nchw_and_nhwc_hold_the_same_values() {
    let Some(g) = gpu() else { return };
    let (src, desc) = nv12_1080p(ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt601 });
    let nchw = letterbox_spec();
    let nhwc = TensorSpec { layout: Layout::Nhwc, ..nchw };
    let a = run(&g, &src, desc, letterbox_m(&desc, &nchw), &nchw);
    let b = run(&g, &src, desc, letterbox_m(&desc, &nhwc), &nhwc);
    assert_eq!(a.out, b.out);
    assert_eq!(compare(&b, 1e-4).1, 0);
}

#[test]
fn f16_matches_f32() {
    let Some(g) = gpu() else { return };
    let (src, desc) = nv12_1080p(ColorSpec::default());
    let f32_spec = TensorSpec { normalize: Normalize::IMAGENET, ..letterbox_spec() };
    let f16_spec = TensorSpec { dtype: DType::F16, ..f32_spec };
    let a = run(&g, &src, desc, letterbox_m(&desc, &f32_spec), &f32_spec);
    let b = run(&g, &src, desc, letterbox_m(&desc, &f16_spec), &f16_spec);
    for (x, y) in a.out.iter().flatten().zip(b.out.iter().flatten()) {
        // Round-to-nearest f16: error <= half an ulp (2^-11 relative).
        assert!((x - y).abs() <= x.abs() * (1.0 / 2048.0) + 1e-7, "{x} vs {y}");
    }
}

#[test]
fn u8_output_is_rounded_pixels() {
    let Some(g) = gpu() else { return };
    let (src, desc) = nv12_1080p(ColorSpec::default());
    let spec = TensorSpec { dtype: DType::U8, layout: Layout::Nhwc, ..letterbox_spec() };
    let r = run(&g, &src, desc, letterbox_m(&desc, &spec), &spec);
    let (max, bad) = compare(&r, 0.0);
    // Exact except for values sitting on a .5 boundary within float noise.
    assert!(max <= 1.0 && bad * 10_000 < r.out[0].len() * 3, "max {max}, {bad} mismatches");
    assert_eq!(r.out[0][0], 114.0);
}

#[test]
fn bgr_is_rgb_swapped() {
    let Some(g) = gpu() else { return };
    let (src, desc) = nv12_1080p(ColorSpec::default());
    let rgb = letterbox_spec();
    let bgr = TensorSpec { color: ColorOrder::Bgr, pad: [1.0, 2.0, 3.0], ..rgb };
    let rgb = TensorSpec { pad: [3.0, 2.0, 1.0], ..rgb };
    let a = run(&g, &src, desc, letterbox_m(&desc, &rgb), &rgb);
    let b = run(&g, &src, desc, letterbox_m(&desc, &bgr), &bgr);
    assert_eq!(a.out[0], b.out[2]);
    assert_eq!(a.out[1], b.out[1]);
    assert_eq!(a.out[2], b.out[0]);
}

#[test]
fn gray_from_yuv_and_rgb() {
    let Some(g) = gpu() else { return };
    let (src, desc) = nv12_1080p(ColorSpec::default());
    let spec =
        TensorSpec { color: ColorOrder::Gray, normalize: Normalize::NONE, ..letterbox_spec() };
    let r = run(&g, &src, desc, letterbox_m(&desc, &spec), &spec);
    assert_eq!(r.out.len(), 1);
    assert_eq!(compare(&r, 1e-3).1, 0);

    let rdesc = FrameDesc::packed(320, 200, PixelFormat::Rgb24, ColorSpec::default());
    let rsrc = pattern(rdesc.byte_len(), 7);
    let spec = TensorSpec { width: 160, height: 160, ..spec };
    let r = run(&g, &rsrc, rdesc, letterbox_m(&rdesc, &spec), &spec);
    assert_eq!(compare(&r, 1e-3).1, 0);
}

#[test]
fn planar_gray_and_bgr_sources_nearest_and_bilinear() {
    let Some(g) = gpu() else { return };
    let formats = [
        PixelFormat::Yuv420p,
        PixelFormat::Yuv422p,
        PixelFormat::Yuv444p,
        PixelFormat::Gray8,
        PixelFormat::Bgr24,
    ];
    for fmt in formats {
        for interp in [Interp::Nearest, Interp::Bilinear] {
            // Odd sizes exercise chroma rounding.
            let desc = FrameDesc::packed(333, 211, fmt, ColorSpec::default());
            let src = pattern(desc.byte_len(), fmt as u32 + 3);
            let spec = TensorSpec {
                width: 128,
                height: 96,
                interp,
                resize: ResizeMode::Stretch,
                ..Default::default()
            };
            let r = run(&g, &src, desc, letterbox_m(&desc, &spec), &spec);
            let (max, bad) = compare(&r, 1e-4);
            assert_eq!(bad, 0, "{fmt:?} {interp:?}: max diff {max}");
        }
    }
}

#[test]
fn affine_rotation_crop() {
    let Some(g) = gpu() else { return };
    let desc = FrameDesc::packed(640, 480, PixelFormat::Rgb24, ColorSpec::default());
    let src = pattern(desc.byte_len(), 11);
    let spec = TensorSpec {
        width: 112,
        height: 112,
        normalize: Normalize::SYMMETRIC,
        pad: [0.0; 3],
        ..Default::default()
    };
    // 30 degree rotation + 0.8 scale around (300, 200), OpenCV convention.
    let (s, c) = (30f32.to_radians().sin() * 0.8, 30f32.to_radians().cos() * 0.8);
    let m = Affine2::from_opencv([
        c,
        -s,
        56.0 - (c * 300.0 - s * 200.0),
        s,
        c,
        56.0 - (s * 300.0 + c * 200.0),
    ]);
    let r = run(&g, &src, desc, m, &spec);
    let (max, bad) = compare(&r, 1e-4);
    assert_eq!(bad, 0, "max diff {max}");
}

#[test]
fn batch_larger_than_one_launch() {
    let Some(g) = gpu() else { return };
    let desc = FrameDesc::packed(256, 256, PixelFormat::Nv12, ColorSpec::default());
    let src = pattern(desc.byte_len(), 5);
    let buf = DeviceBuffer::from_host(&g.device, &src).unwrap();
    let img = SourceImage::from_buffer(&buf, desc).unwrap();
    let spec =
        TensorSpec { width: 32, height: 32, resize: ResizeMode::Stretch, ..Default::default() };
    let n = 70;
    let mats: Vec<Affine2> = (0..n)
        .map(|i| {
            let b = camtensor_core::Roi::Box { x: i as f32 * 2.0, y: i as f32, w: 64.0, h: 48.0 };
            Affine2::sampling_matrix(&b.dst_from_src(32, 32, ResizeMode::Stretch).unwrap()).unwrap()
        })
        .collect();
    let out = DeviceBuffer::zeroed(&g.device, n * spec.image_bytes()).unwrap();
    g.pre.preprocess_into(&g.stream, &img, &mats, &out, 0, &spec).unwrap();
    g.stream.synchronize().unwrap();
    for (i, m) in mats.iter().enumerate() {
        let got = read_output(&out, i * spec.image_bytes(), &spec);
        let px = reference_pixels(&src, &desc, m, &spec);
        for c in 0..3 {
            for p in 0..32 * 32 {
                let e = expected_value(px[c][p], c, &spec);
                assert!((got[c][p] - e).abs() < 1e-4, "item {i} c{c} p{p}: {} vs {e}", got[c][p]);
            }
        }
    }
    // Too-small output buffers are rejected.
    assert!(g.pre.preprocess_into(&g.stream, &img, &mats, &out, 1, &spec).is_err());
}
