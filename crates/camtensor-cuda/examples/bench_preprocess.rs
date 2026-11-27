//! Micro-benchmark of the fused preprocessing kernel.
//!
//! ```text
//! cargo run --release -p camtensor-cuda --example bench_preprocess
//! ```
//!
//! Times (GPU time via CUDA events, averaged over many launches):
//! * 1080p NV12 -> 640x640 f16 NCHW letterbox (YOLO input);
//! * 1080p NV12 -> 640x640 f32 NCHW letterbox;
//! * 1080p NV12 -> 32 x 112x112 f32 ArcFace-style crops in one launch.

use camtensor_core::{
    Affine2, ColorSpec, DType, FrameDesc, Normalize, PixelFormat, Roi, TensorSpec,
};
use camtensor_cuda::{CudaDevice, CudaEvent, CudaStream, DeviceBuffer, Preprocessor, SourceImage};

fn main() -> camtensor_core::Result<()> {
    if !camtensor_cuda::is_available() {
        eprintln!("no CUDA device available");
        return Ok(());
    }
    let device = CudaDevice::get(0)?;
    println!(
        "device: {} (sm_{}{}), driver {:?}",
        device.name(),
        device.compute_capability().0,
        device.compute_capability().1,
        camtensor_cuda::driver_version()
    );
    let stream = CudaStream::new(&device)?;
    let t0 = std::time::Instant::now();
    let pre = Preprocessor::get(&device)?;
    println!("kernel JIT + load: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    let desc = FrameDesc::packed(1920, 1080, PixelFormat::Nv12, ColorSpec::default());
    let host: Vec<u8> = (0..desc.byte_len()).map(|i| (i * 31 % 251) as u8).collect();
    let src_buf = DeviceBuffer::from_host(&device, &host)?;
    let src = SourceImage::from_buffer(&src_buf, desc)?;

    let letterbox = |dtype| TensorSpec { width: 640, height: 640, dtype, ..Default::default() };
    for spec in [letterbox(DType::F16), letterbox(DType::F32)] {
        let m = Affine2::fit(1920.0, 1080.0, 640.0, 640.0, spec.resize);
        let s = Affine2::sampling_matrix(&m).expect("invertible");
        let out = DeviceBuffer::alloc(&device, spec.image_bytes())?;
        let us = time(&device, &stream, 2000, || {
            pre.preprocess_into(&stream, &src, &[s], &out, 0, &spec)
        })?;
        println!(
            "1080p NV12 -> 640x640 {:?} NCHW letterbox: {us:.1} us/frame (kernel)",
            spec.dtype
        );
    }

    let tiny = TensorSpec { width: 8, height: 8, ..Default::default() };
    let tiny_m =
        Affine2::sampling_matrix(&Affine2::fit(1920.0, 1080.0, 8.0, 8.0, tiny.resize)).unwrap();
    let tiny_out = DeviceBuffer::alloc(&device, tiny.image_bytes())?;
    let us = time(&device, &stream, 2000, || {
        pre.preprocess_into(&stream, &src, &[tiny_m], &tiny_out, 0, &tiny)
    })?;
    println!("launch-overhead baseline (8x8 output): {us:.1} us (kernel)");

    let spec = TensorSpec {
        width: 112,
        height: 112,
        normalize: Normalize::SYMMETRIC,
        ..Default::default()
    };
    let crops: Vec<Affine2> = (0..32)
        .map(|i| {
            let x = 100.0 + 50.0 * i as f32;
            let pts = [
                [x + 38.0, 300.0 + 51.0],
                [x + 73.0, 300.0 + 51.0],
                [x + 56.0, 300.0 + 71.0],
                [x + 41.0, 300.0 + 92.0],
                [x + 70.0, 300.0 + 92.0],
            ];
            let m = Roi::Landmarks { points: pts }.dst_from_src(112, 112, spec.resize).unwrap();
            Affine2::sampling_matrix(&m).unwrap()
        })
        .collect();
    let out = DeviceBuffer::alloc(&device, crops.len() * spec.image_bytes())?;
    let us = time(&device, &stream, 2000, || {
        pre.preprocess_into(&stream, &src, &crops, &out, 0, &spec)
    })?;
    println!("1080p NV12 -> 32 x 112x112 f32 aligned crops: {us:.1} us/batch (kernel)");
    Ok(())
}

/// Median kernel execution time of an isolated launch, in microseconds
/// (also prints back-to-back throughput).
fn time(
    device: &std::sync::Arc<CudaDevice>,
    stream: &CudaStream,
    iters: usize,
    mut f: impl FnMut() -> camtensor_core::Result<()>,
) -> camtensor_core::Result<f64> {
    for _ in 0..50 {
        f()?;
    }
    stream.synchronize()?;
    let (start, end) = (CudaEvent::new(device, true)?, CudaEvent::new(device, true)?);
    let wall = std::time::Instant::now();
    start.record(stream)?;
    for _ in 0..iters {
        f()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let stream_us = end.elapsed_ms(&start)? as f64 * 1e3 / iters as f64;
    let wall_us = wall.elapsed().as_secs_f64() * 1e6 / iters as f64;
    // Kernel execution alone: events around single, isolated launches.
    let mut kernel_ms = Vec::with_capacity(200);
    for _ in 0..200 {
        start.record(stream)?;
        f()?;
        end.record(stream)?;
        end.synchronize()?;
        kernel_ms.push(end.elapsed_ms(&start)?);
    }
    kernel_ms.sort_by(f32::total_cmp);
    let kernel_us = kernel_ms[kernel_ms.len() / 2] as f64 * 1e3;
    println!(
        "  back-to-back: {stream_us:.1} us/call on the stream, {wall_us:.1} us/call host wall time"
    );
    Ok(kernel_us)
}
