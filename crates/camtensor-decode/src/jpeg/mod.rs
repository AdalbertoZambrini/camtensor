//! Pure-Rust MJPEG decoder (placeholder until the CPU backend lands).

use camtensor_core::{Device, EncodedPacket, Error, Frame, Result};

use crate::{Decoder, DecoderKind};

/// zune-jpeg based MJPEG decoder.
#[derive(Debug, Default)]
pub struct JpegDecoder;

impl JpegDecoder {
    /// Creates a decoder.
    pub fn new() -> Self {
        Self
    }
}

impl Decoder for JpegDecoder {
    fn decode(&mut self, _: &EncodedPacket, _: bool, _: &mut dyn FnMut(Frame)) -> Result<()> {
        Err(Error::unavailable("jpeg backend not built yet"))
    }
    fn flush(&mut self, _: &mut dyn FnMut(Frame)) -> Result<()> {
        Ok(())
    }
    fn kind(&self) -> DecoderKind {
        DecoderKind::Jpeg
    }
    fn device(&self) -> Device {
        Device::Cpu
    }
}
