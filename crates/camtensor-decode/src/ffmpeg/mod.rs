//! libavcodec decoder (placeholder until the CPU backend lands).

use camtensor_core::{Device, EncodedPacket, Error, Frame, Result, StreamParams};

use crate::{Decoder, DecoderKind};

/// libavcodec software decoder.
#[derive(Debug)]
pub struct FfmpegDecoder;

impl FfmpegDecoder {
    /// Opens a decoder for `params` using `threads` worker threads.
    pub fn open(_params: &StreamParams, _threads: usize) -> Result<Self> {
        Err(Error::unavailable("ffmpeg backend not built yet"))
    }
}

impl Decoder for FfmpegDecoder {
    fn decode(&mut self, _: &EncodedPacket, _: bool, _: &mut dyn FnMut(Frame)) -> Result<()> {
        Err(Error::unavailable("ffmpeg backend not built yet"))
    }
    fn flush(&mut self, _: &mut dyn FnMut(Frame)) -> Result<()> {
        Ok(())
    }
    fn kind(&self) -> DecoderKind {
        DecoderKind::Ffmpeg
    }
    fn device(&self) -> Device {
        Device::Cpu
    }
}
