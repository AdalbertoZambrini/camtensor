//! NVDEC decoder (placeholder until the GPU backend lands).

use camtensor_core::{Device, EncodedPacket, Error, Frame, Result, StreamParams};

use crate::{Decoder, DecoderKind};

/// NVDEC hardware decoder.
#[derive(Debug)]
pub struct NvdecDecoder {
    ordinal: u32,
}

impl NvdecDecoder {
    /// Opens an NVDEC session on CUDA device `ordinal`.
    pub fn open(_params: &StreamParams, ordinal: u32, _surfaces: usize) -> Result<Self> {
        let _ = Self { ordinal };
        Err(Error::unavailable("nvdec backend not built yet"))
    }
}

impl Decoder for NvdecDecoder {
    fn decode(&mut self, _: &EncodedPacket, _: bool, _: &mut dyn FnMut(Frame)) -> Result<()> {
        Err(Error::unavailable("nvdec backend not built yet"))
    }
    fn flush(&mut self, _: &mut dyn FnMut(Frame)) -> Result<()> {
        Ok(())
    }
    fn kind(&self) -> DecoderKind {
        DecoderKind::Nvdec
    }
    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }
}
