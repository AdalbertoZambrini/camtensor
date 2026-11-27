//! Core types shared by every camtensor crate.
//!
//! This crate defines the *output contract* (what a consumer receives), the
//! configuration schema, the encoded-packet and decoded-frame types that flow
//! between the ingest, decode and preprocess stages, and small geometry helpers
//! (letterbox transforms, affine warps, keypoint alignment).
//!
//! It has no heavy dependencies and no `unsafe` code.

#![forbid(unsafe_code)]

pub mod codec;
pub mod config;
pub mod error;
pub mod frame;
pub mod geometry;
pub mod meta;
pub mod tensor;
pub mod time;

pub use codec::{Codec, EncodedPacket, StreamParams};
pub use config::{
    Backpressure, BatchConfig, DecoderPref, EngineConfig, GatingConfig, GatingMode, MediaMtxConfig,
    OutputConfig, StreamConfig, Transport,
};
pub use error::{Error, Result};
pub use frame::{
    ColorMatrix, ColorRange, ColorSpec, DeviceMemory, Frame, FrameData, FrameDesc, PixelFormat,
    PlaneDesc,
};
pub use geometry::{Affine2, KEYPOINT_TEMPLATE_112, Roi};
pub use meta::FrameMeta;
pub use tensor::{
    ColorOrder, DType, Device, Interp, Layout, Normalize, ResizeMode, TensorDesc, TensorSpec,
};
