//! Video decoders for camtensor.
//!
//! All decoders implement [`Decoder`]: they consume [`EncodedPacket`]s
//! (Annex B access units for H.264/H.265, complete JPEG images for MJPEG) and
//! emit [`Frame`]s through a callback.
//!
//! | Backend | Codecs | Output memory | Feature |
//! |---|---|---|---|
//! | NVDEC (`nvcuvid`, loaded at runtime) | H.264, H.265, MJPEG | CUDA device (NV12) | `nvdec` |
//! | libavcodec (loaded at runtime) | H.264, H.265, MJPEG | host (planar YUV) | `ffmpeg` |
//! | zune-jpeg (pure Rust) | MJPEG | host | `jpeg` |
//!
//! No decoder library is linked at build time: missing runtime libraries
//! are reported as [`camtensor_core::Error::Unavailable`] when a decoder is
//! opened, so a single binary runs on machines with or without a GPU.

use camtensor_core::{
    Codec, DecoderPref, Device, EncodedPacket, Error, Frame, Result, StreamParams,
};

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;
#[cfg(feature = "jpeg")]
pub mod jpeg;
#[cfg(feature = "nvdec")]
pub mod nvdec;

/// Identifies a decoder implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecoderKind {
    /// NVIDIA NVDEC.
    Nvdec,
    /// libavcodec software decoder.
    Ffmpeg,
    /// zune-jpeg software decoder.
    Jpeg,
}

impl std::fmt::Display for DecoderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DecoderKind::Nvdec => "nvdec",
            DecoderKind::Ffmpeg => "ffmpeg",
            DecoderKind::Jpeg => "jpeg",
        })
    }
}

/// A video decoder.
///
/// Decoders may reorder (B-frames): a frame emitted while decoding packet `n`
/// can belong to an earlier packet; use [`Frame::pts`] to match frames with
/// packets.
pub trait Decoder: Send {
    /// Decodes one access unit.
    ///
    /// When `emit` is `false` the picture is decoded only to keep the
    /// reference chain intact (gating catch-up); the decoder must not call
    /// `sink` for the frame produced by this packet and should skip any
    /// output copy it can.
    fn decode(
        &mut self,
        pkt: &EncodedPacket,
        emit: bool,
        sink: &mut dyn FnMut(Frame),
    ) -> Result<()>;

    /// Drains frames still buffered in the decoder.
    fn flush(&mut self, sink: &mut dyn FnMut(Frame)) -> Result<()>;

    /// Implementation in use.
    fn kind(&self) -> DecoderKind;

    /// Device where emitted frames live.
    fn device(&self) -> Device;
}

/// Options for [`open`].
#[derive(Debug, Clone)]
pub struct DecoderOptions {
    /// Decoder preference.
    pub pref: DecoderPref,
    /// Target device. With [`Device::Cuda`] NVDEC is preferred; CPU decoders
    /// are used as fall-back and emit host frames (the caller uploads).
    pub device: Device,
    /// Worker threads for CPU decoders (0 = library default).
    pub threads: usize,
    /// Number of NVDEC output surfaces / frame pool size hint.
    pub surfaces: usize,
}

impl Default for DecoderOptions {
    fn default() -> Self {
        Self { pref: DecoderPref::Auto, device: Device::Auto, threads: 0, surfaces: 8 }
    }
}

/// Opens the best available decoder for `params` according to `opts`.
///
/// Resolution order:
/// * `Nvdec` → NVDEC or error.
/// * `Cpu` → libavcodec, then zune-jpeg for MJPEG.
/// * `Auto` → NVDEC when the device is CUDA/Auto and NVDEC is usable, then
///   the CPU decoders.
pub fn open(params: &StreamParams, opts: &DecoderOptions) -> Result<Box<dyn Decoder>> {
    let mut errors: Vec<String> = Vec::new();
    let want_gpu = matches!(opts.device, Device::Cuda(_) | Device::Auto);

    if opts.pref == DecoderPref::Nvdec || (opts.pref == DecoderPref::Auto && want_gpu) {
        #[cfg(feature = "nvdec")]
        {
            let ordinal = match opts.device {
                Device::Cuda(i) => i,
                _ => 0,
            };
            match nvdec::NvdecDecoder::open(params, ordinal, opts.surfaces) {
                Ok(d) => return Ok(Box::new(d)),
                Err(e) => errors.push(format!("nvdec: {e}")),
            }
        }
        #[cfg(not(feature = "nvdec"))]
        errors.push("nvdec: feature disabled".into());
        if opts.pref == DecoderPref::Nvdec {
            return Err(Error::unavailable(errors.join("; ")));
        }
    }

    #[cfg(feature = "ffmpeg")]
    match ffmpeg::FfmpegDecoder::open(params, opts.threads) {
        Ok(d) => return Ok(Box::new(d)),
        Err(e) => errors.push(format!("ffmpeg: {e}")),
    }

    #[cfg(feature = "jpeg")]
    if params.codec == Codec::Mjpeg {
        return Ok(Box::new(jpeg::JpegDecoder::new()));
    }

    let _ = Codec::H264;
    Err(Error::unavailable(format!(
        "no decoder available for {}: {}",
        params.codec,
        errors.join("; ")
    )))
}
