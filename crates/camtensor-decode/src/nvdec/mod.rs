//! NVDEC hardware decoder (H.264, H.265, MJPEG) through the dynamically
//! loaded `nvcuvid` library.
//!
//! Pipeline per packet: `cuvidParseVideoData` → (sequence callback: create
//! or recreate the decoder) → (decode callback: `cuvidDecodePicture`) →
//! (display callback: map the surface, copy it into a pooled, tightly packed
//! device buffer on the decoder's stream, synchronise, unmap). Frames
//! produced during a call are handed to the sink after the parser returns.
//!
//! Emitted frames live in device memory (`FrameData::Device`), are NV12
//! (or YUV 4:4:4 planar when the stream is 4:4:4 and the GPU can output it)
//! and are fully written when emitted. Buffers come from a pool and are
//! reused once every clone of the frame has been dropped.

use std::collections::VecDeque;
use std::sync::Arc;

use camtensor_core::{
    Codec, ColorMatrix, ColorRange, ColorSpec, Device, EncodedPacket, Error, Frame, FrameData,
    FrameDesc, PixelFormat, Result, StreamParams,
};
use camtensor_cuda::nvdec::{
    self as nv, CUVIDEOFORMAT, CUVIDPARSERDISPINFO, DecoderConfig, PacketFlags, ParserHandler,
    PictureParams, Rect, VideoDecoder, VideoParser,
};
use camtensor_cuda::{BufferPool, Copy2d, CudaDevice, CudaStream};

use crate::{Decoder, DecoderKind};

/// Timestamp clock given to the parser (camtensor PTS are 90 kHz).
const CLOCK_RATE: u32 = 90_000;
/// Max number of pending suppressed timestamps remembered.
const MAX_SUPPRESSED: usize = 64;

/// NVDEC hardware decoder.
pub struct NvdecDecoder {
    parser: Option<VideoParser<State>>,
    cuvid_codec: nv::cudaVideoCodec,
    device: Arc<CudaDevice>,
}

impl std::fmt::Debug for NvdecDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvdecDecoder")
            .field("device", &self.device.ordinal())
            .field("codec", &self.cuvid_codec)
            .finish_non_exhaustive()
    }
}

/// Output layout of the current sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Output {
    width: u32,
    height: u32,
    /// Rows of the mapped surface's luma plane (offset of the next plane is
    /// `pitch * surface_height`).
    surface_height: u32,
    format: PixelFormat,
    color: ColorSpec,
}

struct State {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    codec: Codec,
    cuvid_codec: nv::cudaVideoCodec,
    decoder: Option<VideoDecoder>,
    output: Option<Output>,
    pool: Option<BufferPool>,
    pool_size: usize,
    suppressed: VecDeque<i64>,
    pending: Vec<Frame>,
    error: Option<Error>,
}

impl NvdecDecoder {
    /// Opens an NVDEC session on CUDA device `ordinal`. `surfaces` is the
    /// size of the output frame pool (frames the consumer may hold at once
    /// before extra buffers are allocated).
    ///
    /// Returns [`Error::Unavailable`] when the driver, NVDEC or codec support
    /// is missing, so callers can fall back to a CPU decoder.
    pub fn open(params: &StreamParams, ordinal: u32, surfaces: usize) -> Result<Self> {
        nv::api()?;
        let device = CudaDevice::get(ordinal)?;
        let cuvid_codec = match params.codec {
            Codec::H264 => nv::cudaVideoCodec_H264,
            Codec::H265 => nv::cudaVideoCodec_HEVC,
            Codec::Mjpeg => nv::cudaVideoCodec_JPEG,
            Codec::RawVideo => return Err(Error::unavailable("raw video needs no decoder")),
        };
        let caps = nv::decoder_caps(&device, cuvid_codec, nv::cudaVideoChromaFormat_420, 0)?;
        if !caps.supported {
            return Err(Error::unavailable(format!(
                "NVDEC on {} does not support {} 8-bit 4:2:0",
                device.name(),
                params.codec
            )));
        }
        if params.width > 0 && params.height > 0 && !caps.fits(params.width, params.height) {
            return Err(Error::unavailable(format!(
                "NVDEC cannot decode {}x{} {} (max {}x{})",
                params.width, params.height, params.codec, caps.max_width, caps.max_height
            )));
        }
        let stream = CudaStream::new(&device)?;
        let state = State {
            device: device.clone(),
            stream,
            codec: params.codec,
            cuvid_codec,
            decoder: None,
            output: None,
            pool: None,
            pool_size: surfaces.max(2),
            suppressed: VecDeque::new(),
            pending: Vec::new(),
            error: None,
        };
        let parser = VideoParser::new(cuvid_codec, 0, CLOCK_RATE, state)?;
        Ok(Self { parser: Some(parser), cuvid_codec, device })
    }

    fn parser(&mut self) -> Result<&mut VideoParser<State>> {
        self.parser.as_mut().ok_or_else(|| Error::decode("NVDEC parser lost after a failed reset"))
    }

    /// Hands pending frames to `sink`; returns the first callback error.
    fn finish(&mut self, parse: Result<()>, sink: &mut dyn FnMut(Frame)) -> Result<()> {
        let state = self.parser()?.handler_mut();
        let frames = std::mem::take(&mut state.pending);
        let err = state.error.take();
        for f in frames {
            sink(f);
        }
        match err {
            Some(e) => Err(e),
            None => parse,
        }
    }
}

impl Decoder for NvdecDecoder {
    fn decode(
        &mut self,
        pkt: &EncodedPacket,
        emit: bool,
        sink: &mut dyn FnMut(Frame),
    ) -> Result<()> {
        if pkt.data.is_empty() {
            return Ok(());
        }
        let device = self.device.clone();
        let _ctx = device.bind()?;
        let parser = self.parser()?;
        if !emit {
            let st = parser.handler_mut();
            if st.suppressed.len() >= MAX_SUPPRESSED {
                st.suppressed.pop_front();
            }
            st.suppressed.push_back(pkt.pts);
        }
        let flags = PacketFlags { end_of_picture: true, discontinuity: false };
        let r = parser.parse(&pkt.data, Some(pkt.pts), flags);
        self.finish(r, sink)
    }

    fn flush(&mut self, sink: &mut dyn FnMut(Frame)) -> Result<()> {
        let device = self.device.clone();
        let _ctx = device.bind()?;
        let r = self.parser()?.end_of_stream();
        let r = self.finish(r, sink);
        // A parser does not accept data after end of stream: start a fresh
        // one, keeping the decoder (it is reused if the format matches).
        let mut state = self.parser.take().map(VideoParser::into_handler);
        if let Some(st) = state.as_mut() {
            st.suppressed.clear();
        }
        if let Some(st) = state {
            self.parser = Some(VideoParser::new(self.cuvid_codec, 0, CLOCK_RATE, st)?);
        }
        r
    }

    fn kind(&self) -> DecoderKind {
        DecoderKind::Nvdec
    }

    fn device(&self) -> Device {
        Device::Cuda(self.device.ordinal())
    }
}

/// Colour description of a sequence.
fn color_spec(codec: Codec, f: &CUVIDEOFORMAT) -> ColorSpec {
    if codec == Codec::Mjpeg {
        return ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt601 };
    }
    let sig = f.video_signal_description;
    let range = if sig.full_range() { ColorRange::Full } else { ColorRange::Limited };
    let height = (f.display_area.bottom - f.display_area.top).max(0) as u32;
    let matrix = match sig.matrix_coefficients {
        1 => ColorMatrix::Bt709,
        5 | 6 => ColorMatrix::Bt601,
        _ if height >= 720 => ColorMatrix::Bt709,
        _ => ColorMatrix::Bt601,
    };
    ColorSpec { range, matrix }
}

impl State {
    fn on_sequence(&mut self, f: &CUVIDEOFORMAT) -> Result<i32> {
        if f.bit_depth_luma_minus8 > 0 || f.bit_depth_chroma_minus8 > 0 {
            return Err(Error::decode(format!(
                "unsupported: {}-bit {} (the NVDEC backend handles 8-bit only for now)",
                f.bit_depth_luma_minus8 + 8,
                self.codec
            )));
        }
        let chroma = f.chroma_format;
        let caps = nv::decoder_caps(&self.device, self.cuvid_codec, chroma, 0)?;
        let chroma_name = match chroma {
            nv::cudaVideoChromaFormat_Monochrome => "4:0:0",
            nv::cudaVideoChromaFormat_420 => "4:2:0",
            nv::cudaVideoChromaFormat_422 => "4:2:2",
            _ => "4:4:4",
        };
        if !caps.supported {
            return Err(Error::decode(format!(
                "unsupported: {} {chroma_name} is not supported by NVDEC on {}",
                self.codec,
                self.device.name()
            )));
        }
        if !caps.fits(f.coded_width, f.coded_height) {
            return Err(Error::decode(format!(
                "unsupported: {}x{} {} exceeds NVDEC limits ({}x{})",
                f.coded_width, f.coded_height, self.codec, caps.max_width, caps.max_height
            )));
        }
        // Surface format: NV12 for 4:0:0 / 4:2:0 (and 4:2:2 when the driver
        // can downsample), planar 4:4:4 for 4:4:4.
        let (surface, format) = if chroma == nv::cudaVideoChromaFormat_444
            && caps.supports_output(nv::cudaVideoSurfaceFormat_YUV444)
        {
            (nv::cudaVideoSurfaceFormat_YUV444, PixelFormat::Yuv444p)
        } else if caps.supports_output(nv::cudaVideoSurfaceFormat_NV12)
            && chroma != nv::cudaVideoChromaFormat_444
        {
            (nv::cudaVideoSurfaceFormat_NV12, PixelFormat::Nv12)
        } else {
            return Err(Error::decode(format!(
                "unsupported: no 8-bit output surface for {} {chroma_name} (mask {:#x})",
                self.codec, caps.output_format_mask
            )));
        };

        let mut area = Rect {
            left: f.display_area.left,
            top: f.display_area.top,
            right: f.display_area.right,
            bottom: f.display_area.bottom,
        };
        if area.right <= area.left || area.bottom <= area.top {
            area = Rect {
                left: 0,
                top: 0,
                right: f.coded_width as i32,
                bottom: f.coded_height as i32,
            };
        }
        let width = (area.right - area.left) as u32;
        let height = (area.bottom - area.top) as u32;
        let target_w = (width + 1) & !1;
        let target_h = (height + 1) & !1;
        let surfaces = (f.min_num_decode_surfaces as u32).max(1);
        let config = DecoderConfig {
            codec: self.cuvid_codec,
            chroma_format: chroma,
            bit_depth_minus8: 0,
            coded_width: f.coded_width,
            coded_height: f.coded_height,
            num_decode_surfaces: surfaces,
            num_output_surfaces: 2,
            output_format: surface,
            display_area: area,
            target_width: target_w,
            target_height: target_h,
            deinterlace: f.progressive_sequence == 0 && self.codec != Codec::Mjpeg,
        };
        let output = Output {
            width,
            height,
            surface_height: target_h,
            format,
            color: color_spec(self.codec, f),
        };
        if self.decoder.as_ref().map(|d| *d.config()) != Some(config) {
            // Destroy first: NVDEC sessions are a limited resource.
            self.decoder = None;
            self.decoder = Some(VideoDecoder::new(&self.device, &config)?);
            tracing::debug!(
                codec = %self.codec,
                width,
                height,
                surfaces,
                ?format,
                "NVDEC decoder created"
            );
        }
        let byte_len = FrameDesc::packed(width, height, format, output.color).byte_len();
        if self.pool.as_ref().map(BufferPool::byte_len) != Some(byte_len) {
            self.pool = Some(BufferPool::new(&self.device, byte_len, self.pool_size));
        }
        self.output = Some(output);
        Ok(surfaces as i32)
    }

    fn on_display(&mut self, info: &CUVIDPARSERDISPINFO) -> Result<()> {
        if let Some(i) = self.suppressed.iter().position(|&p| p == info.timestamp) {
            self.suppressed.remove(i);
            return Ok(());
        }
        let (Some(dec), Some(out), Some(pool)) = (&self.decoder, self.output, &self.pool) else {
            return Err(Error::decode("NVDEC display callback before sequence header"));
        };
        let desc = FrameDesc::packed(out.width, out.height, out.format, out.color);
        let buf = pool.get()?;
        {
            let mapped = dec.map(info, &self.stream)?;
            if self.codec == Codec::Mjpeg {
                // NVDEC JPEG post-processing runs on the legacy default
                // stream, which our non-blocking stream does not wait for.
                self.device.synchronize_default_stream()?;
            }
            let (src, pitch) = (mapped.device_ptr(), mapped.pitch());
            let plane_src = pitch as u64 * out.surface_height as u64;
            let (w, h) = (out.width as usize, out.height as usize);
            let copies: Vec<(u64, usize, usize, usize)> = match out.format {
                PixelFormat::Nv12 => {
                    vec![(src, 0, w, h), (src + plane_src, 1, w.div_ceil(2) * 2, h.div_ceil(2))]
                }
                _ => (0..3).map(|i| (src + plane_src * i as u64, i, w, h)).collect(),
            };
            for (plane_ptr, i, row, rows) in copies {
                let p = desc.planes[i];
                // SAFETY: the mapped surface holds `rows` rows of `pitch`
                // bytes per plane (planes `surface_height` rows apart), and
                // the pooled buffer was sized for `desc`; the copy completes
                // before `mapped` is unmapped (synchronize below).
                unsafe {
                    self.stream.memcpy_2d_async(&Copy2d {
                        src: plane_ptr,
                        src_host: false,
                        src_pitch: pitch,
                        dst: buf.device_ptr() + p.offset as u64,
                        dst_host: false,
                        dst_pitch: p.pitch,
                        width_bytes: row,
                        height: rows,
                    })?;
                }
            }
            self.stream.synchronize()?;
        }
        if dec.had_error(info.picture_index) == Some(true) {
            tracing::debug!(pts = info.timestamp, "NVDEC reported a (concealed) decode error");
        }
        self.pending.push(Frame { desc, data: FrameData::Device(buf), pts: info.timestamp });
        Ok(())
    }

    fn fail(&mut self, e: Error) -> i32 {
        if self.error.is_none() {
            self.error = Some(e);
        }
        0
    }
}

impl ParserHandler for State {
    fn sequence(&mut self, format: &CUVIDEOFORMAT) -> i32 {
        match self.on_sequence(format) {
            Ok(n) => n,
            Err(e) => self.fail(e),
        }
    }

    fn decode(&mut self, pic: &mut PictureParams<'_>) -> i32 {
        let Some(dec) = &self.decoder else {
            return self.fail(Error::decode("NVDEC picture before sequence header"));
        };
        match dec.decode(pic) {
            Ok(()) => 1,
            Err(e) => self.fail(e),
        }
    }

    fn display(&mut self, info: Option<&CUVIDPARSERDISPINFO>) -> i32 {
        let Some(info) = info else { return 1 };
        match self.on_display(info) {
            Ok(()) => 1,
            Err(e) => self.fail(e),
        }
    }
}
