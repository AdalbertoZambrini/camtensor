//! `cargo bench -p camtensor-preprocess`

use std::hint::black_box;
use std::time::Duration;

use camtensor_core::{
    ARCFACE_112_TEMPLATE, Affine2, ColorSpec, DType, FrameDesc, Normalize, PixelFormat, Roi,
    TensorSpec,
};
use camtensor_preprocess::{Parallelism, plan, plan_roi, preprocess_into, preprocess_into_with};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// Smooth-ish synthetic frame (gradients + noise) so caches behave as with
/// real video.
fn frame(w: u32, h: u32, format: PixelFormat) -> (FrameDesc, Vec<u8>) {
    let desc = FrameDesc::packed(w, h, format, ColorSpec::default());
    let mut seed = 0x1234_5678u32;
    let data = (0..desc.byte_len())
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ((i % 251) as u32 + (seed & 15)) as u8
        })
        .collect();
    (desc, data)
}

fn full_frame(c: &mut Criterion, name: &str, src: (u32, u32, PixelFormat), spec: TensorSpec) {
    let (desc, data) = frame(src.0, src.1, src.2);
    let p = plan(src.0, src.1, &spec);
    let mut out = vec![0u8; spec.image_bytes()];
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(1));
    g.bench_function("single", |b| {
        b.iter(|| preprocess_into(&desc, black_box(&data), &p.sampling, &spec, &mut out).unwrap())
    });
    g.bench_function("rayon", |b| {
        b.iter(|| {
            preprocess_into_with(
                &desc,
                black_box(&data),
                &p.sampling,
                &spec,
                &mut out,
                Parallelism::Rayon,
            )
            .unwrap()
        })
    });
    g.finish();
}

fn benches(c: &mut Criterion) {
    let yolo = TensorSpec::default();
    full_frame(
        c,
        "nv12_1080p_to_640_f16_nchw_letterbox",
        (1920, 1080, PixelFormat::Nv12),
        TensorSpec { dtype: DType::F16, ..yolo },
    );
    full_frame(c, "nv12_1080p_to_640_f32_nchw_letterbox", (1920, 1080, PixelFormat::Nv12), yolo);
    full_frame(
        c,
        "yuv420p_1080p_to_640_f32_nchw_letterbox",
        (1920, 1080, PixelFormat::Yuv420p),
        yolo,
    );
    full_frame(
        c,
        "nv12_4k_to_640_f16_nchw_letterbox",
        (3840, 2160, PixelFormat::Nv12),
        TensorSpec { dtype: DType::F16, ..yolo },
    );

    // 16 ArcFace crops (planning included) from a 1080p NV12 frame.
    let (desc, data) = frame(1920, 1080, PixelFormat::Nv12);
    let spec = TensorSpec {
        width: 112,
        height: 112,
        normalize: Normalize::SYMMETRIC,
        ..TensorSpec::default()
    };
    let faces: Vec<Roi> = (0..16)
        .map(|i| {
            let (t, s) = (0.05 * (i as f32 - 8.0), 1.5 + 0.1 * i as f32);
            let (cx, cy) = (200.0 + 100.0 * i as f32, 300.0 + 30.0 * i as f32);
            let rot = Affine2([s * t.cos(), -s * t.sin(), cx, s * t.sin(), s * t.cos(), cy]);
            let mut points = ARCFACE_112_TEMPLATE;
            for p in &mut points {
                let (x, y) = rot.apply(p[0] - 56.0, p[1] - 56.0);
                *p = [x, y];
            }
            Roi::Landmarks { points }
        })
        .collect();
    let mut out = vec![0u8; spec.image_bytes() * faces.len()];
    let mut g = c.benchmark_group("arcface_16x112_f32_from_nv12_1080p");
    g.throughput(Throughput::Elements(faces.len() as u64));
    g.bench_function("single", |b| {
        b.iter(|| {
            for (roi, o) in faces.iter().zip(out.chunks_exact_mut(spec.image_bytes())) {
                let p = plan_roi(roi, &spec).unwrap();
                preprocess_into(&desc, black_box(&data), &p.sampling, &spec, o).unwrap();
            }
        })
    });
    g.finish();
}

criterion_group! {
    name = group;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30);
    targets = benches
}
criterion_main!(group);
