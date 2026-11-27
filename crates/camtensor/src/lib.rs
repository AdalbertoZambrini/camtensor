//! # camtensor
//!
//! Camera streams to inference-ready tensors.
//!
//! camtensor connects to RTSP cameras (directly or through MediaMTX),
//! decodes them on NVDEC or the CPU, and writes each frame **once**, already
//! colour-converted, resized/letterboxed, normalised and laid out
//! (NCHW/NHWC, f32/f16/u8), straight into pooled batch tensors on the GPU or
//! in host memory. Optional compressed-domain motion gating skips decoding
//! static scenes, and the crop API turns detector keypoints into aligned
//! crops (boxes, affine warps, keypoint alignment) without leaving the GPU.
//!
//! ```no_run
//! use camtensor::{Engine, EngineConfig};
//!
//! let engine = Engine::new(EngineConfig::from_yaml(r#"
//! device: auto
//! outputs:
//!   detector:
//!     tensor: { width: 640, height: 640, dtype: f16, resize: letterbox, normalize: unit }
//!     batch: { max_size: 8, max_wait_ms: 15 }
//!     keep_frame: true
//! streams:
//!   - { id: cam1, url: "rtsp://127.0.0.1:8554/cam1", outputs: [detector], gating: { mode: auto } }
//! "#)?)?;
//!
//! for batch in engine.subscribe("detector")? {
//!     let t = batch.tensor(); // [N, 3, 640, 640] f16 on cuda:0 (or CPU)
//!     for item in batch.items() {
//!         let _box_in_source = item.meta.box_to_source([10.0, 20.0, 110.0, 220.0]);
//!     }
//!     # let _ = t;
//! }
//! # Ok::<(), camtensor::Error>(())
//! ```

#![warn(missing_docs)]

mod device;
mod engine;
mod mediamtx;
mod output;
mod stats;
mod stream;
mod tensor;

pub use camtensor_core::{
    self as core, Affine2, Backpressure, BatchConfig, Codec, ColorOrder, DType, DecoderPref,
    Device, EngineConfig, Error, Frame, FrameMeta, GatingConfig, GatingMode, Interp, Layout,
    MediaMtxConfig, Normalize, OutputConfig, ResizeMode, Result, Roi, StreamConfig, TensorDesc,
    TensorSpec, Transport,
};
pub use engine::{Engine, Subscription};
pub use output::OutputStats;
pub use stats::{EngineStats, StreamStats};
pub use tensor::{AlignedBuf, Batch, BatchItem, Tensor, TensorStorage};

/// Version of the camtensor crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Reports which optional backends are usable on this machine.
pub fn capabilities() -> Capabilities {
    Capabilities {
        cuda: cfg!(feature = "cuda") && cuda_available(),
        ffmpeg: cfg!(feature = "ffmpeg"),
        jpeg: cfg!(feature = "jpeg"),
    }
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    camtensor_cuda::is_available()
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

/// Backends compiled in / available at runtime.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Capabilities {
    /// CUDA driver and NVDEC usable.
    pub cuda: bool,
    /// libavcodec support compiled in (the library is loaded on first use).
    pub ffmpeg: bool,
    /// Pure-Rust MJPEG decoder compiled in.
    pub jpeg: bool,
}
