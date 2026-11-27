//! camtensor: camera streams to inference-ready tensors.

mod output;
mod tensor;

pub use output::OutputStats;
pub use tensor::{AlignedBuf, Batch, BatchItem, Tensor, TensorStorage};
