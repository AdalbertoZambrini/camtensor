//! Safe wrappers around the NVDEC parser and decoder objects.
//!
//! The decoding *policy* (surface management, frame output) lives in
//! `camtensor-decode`; this module only makes the cuvid objects safe to use:
//! RAII handles, C callbacks that never unwind, and parser-provided picture
//! parameters that cannot be forged.

use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_ulong, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, OnceLock};

use camtensor_core::{Error, Result};

use crate::device::CudaDevice;
use crate::error::{check, error};
use crate::stream::CudaStream;
use crate::sys::cuvid::{self as sys, CuvidApi};

pub use crate::sys::cuvid::{
    CUVIDEOFORMAT, CUVIDPARSERDISPINFO, cudaVideoChromaFormat, cudaVideoChromaFormat_420,
    cudaVideoChromaFormat_422, cudaVideoChromaFormat_444, cudaVideoChromaFormat_Monochrome,
    cudaVideoCodec, cudaVideoCodec_H264, cudaVideoCodec_HEVC, cudaVideoCodec_JPEG,
    cudaVideoSurfaceFormat, cudaVideoSurfaceFormat_NV12, cudaVideoSurfaceFormat_P016,
    cudaVideoSurfaceFormat_YUV444,
};

/// NVDEC function table or `Unavailable`.
pub fn api() -> Result<&'static CuvidApi> {
    sys::api().map_err(|e| Error::unavailable(format!("NVDEC (nvcuvid) not available: {e}")))
}

/// `true` when both the CUDA driver and nvcuvid load and a device exists.
pub fn is_available() -> bool {
    crate::is_available() && sys::api().is_ok()
}

/// Output of `cuvidGetDecoderCaps`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecoderCaps {
    /// The codec / chroma / bit-depth combination is supported.
    pub supported: bool,
    /// Bit `i` set when `cudaVideoSurfaceFormat` `i` can be output.
    pub output_format_mask: u16,
    /// Maximum coded width.
    pub max_width: u32,
    /// Maximum coded height.
    pub max_height: u32,
    /// Maximum macroblock count (`w * h / 256`).
    pub max_mb_count: u32,
    /// Minimum coded width.
    pub min_width: u32,
    /// Minimum coded height.
    pub min_height: u32,
}

impl DecoderCaps {
    /// `true` when `format` is in the output mask.
    pub fn supports_output(&self, format: cudaVideoSurfaceFormat) -> bool {
        (0..16).contains(&format) && self.output_format_mask & (1 << format) != 0
    }

    /// `true` when a `width x height` stream fits the limits.
    pub fn fits(&self, width: u32, height: u32) -> bool {
        width >= self.min_width
            && height >= self.min_height
            && width <= self.max_width
            && height <= self.max_height
            && (width as u64 * height as u64) / 256 <= self.max_mb_count as u64
    }
}

/// Queries NVDEC capabilities on `device`.
pub fn decoder_caps(
    device: &CudaDevice,
    codec: cudaVideoCodec,
    chroma: cudaVideoChromaFormat,
    bit_depth_minus8: u32,
) -> Result<DecoderCaps> {
    let api = api()?;
    let _g = device.bind()?;
    let mut caps = sys::CUVIDDECODECAPS {
        eCodecType: codec,
        eChromaFormat: chroma,
        nBitDepthMinus8: bit_depth_minus8,
        ..Default::default()
    };
    // SAFETY: context current; `caps` is a valid, initialised struct.
    check(unsafe { (api.cuvidGetDecoderCaps)(&mut caps) }, "cuvidGetDecoderCaps")?;
    Ok(DecoderCaps {
        supported: caps.bIsSupported != 0,
        output_format_mask: caps.nOutputFormatMask,
        max_width: caps.nMaxWidth,
        max_height: caps.nMaxHeight,
        max_mb_count: caps.nMaxMBCount,
        min_width: caps.nMinWidth as u32,
        min_height: caps.nMinHeight as u32,
    })
}

/// Rectangle in pixels (`right`/`bottom` exclusive).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rect {
    /// Left edge.
    pub left: i32,
    /// Top edge.
    pub top: i32,
    /// Right edge (exclusive).
    pub right: i32,
    /// Bottom edge (exclusive).
    pub bottom: i32,
}

/// Parameters of [`VideoDecoder::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderConfig {
    /// Codec.
    pub codec: cudaVideoCodec,
    /// Chroma format of the bitstream.
    pub chroma_format: cudaVideoChromaFormat,
    /// Bit depth minus 8.
    pub bit_depth_minus8: u32,
    /// Coded width.
    pub coded_width: u32,
    /// Coded height.
    pub coded_height: u32,
    /// Number of decode surfaces (>= the parser's `min_num_decode_surfaces`).
    pub num_decode_surfaces: u32,
    /// Maximum number of simultaneously mapped output surfaces.
    pub num_output_surfaces: u32,
    /// Output surface format.
    pub output_format: cudaVideoSurfaceFormat,
    /// Source crop (display area).
    pub display_area: Rect,
    /// Output (post-processed) width; must be even.
    pub target_width: u32,
    /// Output (post-processed) height; must be even.
    pub target_height: u32,
    /// Use adaptive deinterlacing (otherwise weave).
    pub deinterlace: bool,
}

/// Process-wide NVDEC context lock of each device (`cuvidCtxLockCreate`),
/// shared by all decoders on that device. The driver takes it around the
/// CUDA work it does on our behalf, which makes concurrent decoders on
/// different threads safe. Created once and never destroyed.
fn context_lock(device: &CudaDevice) -> Result<sys::CUvideoctxlock> {
    static LOCKS: OnceLock<Mutex<HashMap<u32, usize>>> = OnceLock::new();
    let api = api()?;
    let mut map = LOCKS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&l) = map.get(&device.ordinal()) {
        return Ok(l as sys::CUvideoctxlock);
    }
    let mut lock: sys::CUvideoctxlock = std::ptr::null_mut();
    check(
        // SAFETY: valid out pointer and retained primary context.
        unsafe { (api.cuvidCtxLockCreate)(&mut lock, device.raw_context()) },
        "cuvidCtxLockCreate",
    )?;
    map.insert(device.ordinal(), lock as usize);
    Ok(lock)
}

/// An NVDEC decoder instance (`cuvidCreateDecoder`).
pub struct VideoDecoder {
    raw: sys::CUvideodecoder,
    device: Arc<CudaDevice>,
    config: DecoderConfig,
}

// SAFETY: the decoder handle is not tied to a thread; every call binds the
// owning context. It is not `Sync`: cuvid decoder calls are not
// synchronised internally without a context lock.
unsafe impl Send for VideoDecoder {}

impl std::fmt::Debug for VideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoDecoder").field("config", &self.config).finish()
    }
}

impl VideoDecoder {
    /// Creates a decoder on `device`.
    pub fn new(device: &Arc<CudaDevice>, config: &DecoderConfig) -> Result<Self> {
        let api = api()?;
        let rect16 = |v: i32| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let mut info = sys::CUVIDDECODECREATEINFO {
            ulWidth: config.coded_width as c_ulong,
            ulHeight: config.coded_height as c_ulong,
            ulNumDecodeSurfaces: config.num_decode_surfaces as c_ulong,
            CodecType: config.codec,
            ChromaFormat: config.chroma_format,
            ulCreationFlags: sys::cudaVideoCreate_PreferCUVID,
            bitDepthMinus8: config.bit_depth_minus8 as c_ulong,
            ulMaxWidth: config.coded_width as c_ulong,
            ulMaxHeight: config.coded_height as c_ulong,
            display_area: sys::CUVIDRECT16 {
                left: rect16(config.display_area.left),
                top: rect16(config.display_area.top),
                right: rect16(config.display_area.right),
                bottom: rect16(config.display_area.bottom),
            },
            OutputFormat: config.output_format,
            DeinterlaceMode: if config.deinterlace {
                sys::cudaVideoDeinterlaceMode_Adaptive
            } else {
                sys::cudaVideoDeinterlaceMode_Weave
            },
            ulTargetWidth: config.target_width as c_ulong,
            ulTargetHeight: config.target_height as c_ulong,
            ulNumOutputSurfaces: config.num_output_surfaces.max(1) as c_ulong,
            ..Default::default()
        };
        info.vidLock = context_lock(device)?;
        let _g = device.bind()?;
        let mut raw: sys::CUvideodecoder = std::ptr::null_mut();
        // SAFETY: context current; `info` is fully initialised with the
        // device's context lock (never destroyed); valid out pointer.
        check(unsafe { (api.cuvidCreateDecoder)(&mut raw, &mut info) }, "cuvidCreateDecoder")?;
        Ok(Self { raw, device: device.clone(), config: *config })
    }

    /// Configuration the decoder was created with.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Submits a picture for decoding (`cuvidDecodePicture`).
    pub fn decode(&self, pic: &mut PictureParams<'_>) -> Result<()> {
        let api = api()?;
        let _g = self.device.bind()?;
        // SAFETY: `pic` was produced by the parser for this call chain, so
        // its bitstream pointers are valid until the callback returns.
        check(unsafe { (api.cuvidDecodePicture)(self.raw, pic.raw) }, "cuvidDecodePicture")
    }

    /// Decode status of picture `index`: `Some(true)` when an error was
    /// reported (concealed or not), `None` if the query is unsupported.
    pub fn had_error(&self, index: i32) -> Option<bool> {
        let api = api().ok()?;
        let _g = self.device.bind().ok()?;
        let mut st = sys::CUVIDGETDECODESTATUS::default();
        // SAFETY: valid decoder; out struct initialised.
        let r = unsafe { (api.cuvidGetDecodeStatus)(self.raw, index as c_int, &mut st) };
        if r != 0 {
            return None;
        }
        Some(
            st.decodeStatus == sys::cuvidDecodeStatus_Error
                || st.decodeStatus == sys::cuvidDecodeStatus_Error_Concealed,
        )
    }

    /// Maps the decoded surface for `info` (post-processing runs on
    /// `stream`). Unmapped when the returned guard is dropped.
    pub fn map(&self, info: &CUVIDPARSERDISPINFO, stream: &CudaStream) -> Result<MappedFrame<'_>> {
        let api = api()?;
        let _g = self.device.bind()?;
        let mut proc = sys::CUVIDPROCPARAMS {
            progressive_frame: info.progressive_frame,
            second_field: info.repeat_first_field + 1,
            top_field_first: info.top_field_first,
            unpaired_field: (info.repeat_first_field < 0) as c_int,
            output_stream: stream.raw(),
            ..Default::default()
        };
        let mut ptr: u64 = 0;
        let mut pitch: c_uint = 0;
        check(
            // SAFETY: context current; valid decoder and out pointers; the
            // driver validates the picture index.
            unsafe {
                (api.cuvidMapVideoFrame64)(
                    self.raw,
                    info.picture_index,
                    &mut ptr,
                    &mut pitch,
                    &mut proc,
                )
            },
            "cuvidMapVideoFrame64",
        )?;
        Ok(MappedFrame { decoder: self, ptr, pitch: pitch as usize })
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        if let (Ok(api), Ok(_g)) = (api(), self.device.bind()) {
            // SAFETY: created by us, destroyed once; no frame is mapped
            // because `MappedFrame` borrows the decoder.
            unsafe { (api.cuvidDestroyDecoder)(self.raw) };
        }
    }
}

/// A mapped output surface; unmapped on drop.
#[derive(Debug)]
pub struct MappedFrame<'a> {
    decoder: &'a VideoDecoder,
    ptr: u64,
    pitch: usize,
}

impl MappedFrame<'_> {
    /// Device pointer of the surface (luma plane first).
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Row pitch in bytes.
    pub fn pitch(&self) -> usize {
        self.pitch
    }
}

impl Drop for MappedFrame<'_> {
    fn drop(&mut self) {
        if let (Ok(api), Ok(_g)) = (api(), self.decoder.device.bind()) {
            // SAFETY: the pointer was returned by cuvidMapVideoFrame64 on this
            // decoder and is unmapped exactly once.
            unsafe { (api.cuvidUnmapVideoFrame64)(self.decoder.raw, self.ptr) };
        }
    }
}

/// Picture parameters handed to [`ParserHandler::decode`]. Only the parser
/// can create them, which guarantees the embedded pointers are valid.
#[derive(Debug)]
pub struct PictureParams<'a> {
    raw: &'a mut sys::CUVIDPICPARAMS,
}

impl PictureParams<'_> {
    /// Decode surface index of the picture.
    pub fn picture_index(&self) -> i32 {
        self.raw.CurrPicIdx
    }

    /// `true` for intra-only pictures.
    pub fn is_intra(&self) -> bool {
        self.raw.intra_pic_flag != 0
    }

    /// `true` if the picture is used as a reference.
    pub fn is_reference(&self) -> bool {
        self.raw.ref_pic_flag != 0
    }
}

/// Callbacks of a [`VideoParser`]. Return values follow the cuvid
/// convention (`0` = failure; for `sequence`, a value `> 1` overrides the
/// parser's DPB size).
///
/// Panics are caught and turned into an error of the current
/// [`VideoParser::parse`] call.
pub trait ParserHandler {
    /// New or changed sequence header.
    fn sequence(&mut self, format: &CUVIDEOFORMAT) -> i32;
    /// A picture is ready to be decoded (decode order).
    fn decode(&mut self, pic: &mut PictureParams<'_>) -> i32;
    /// A picture is ready to be displayed (display order); `None` marks the
    /// end of stream when the parser signals it.
    fn display(&mut self, info: Option<&CUVIDPARSERDISPINFO>) -> i32;
}

struct ParserInner<H> {
    handler: H,
    panic: Option<String>,
}

/// An NVDEC bitstream parser (`cuvidCreateVideoParser`) driving a
/// [`ParserHandler`].
pub struct VideoParser<H: ParserHandler> {
    raw: sys::CUvideoparser,
    /// Owned `Box<ParserInner<H>>`, kept as a raw pointer because the
    /// parser aliases it through `pUserData` during callbacks.
    inner: *mut ParserInner<H>,
}

// SAFETY: the parser may be used from any thread, one at a time (`&mut`
// for parsing); the handler moves with it.
unsafe impl<H: ParserHandler + Send> Send for VideoParser<H> {}

impl<H: ParserHandler> std::fmt::Debug for VideoParser<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoParser").field("raw", &self.raw).finish()
    }
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Runs `f` on the handler behind `user`, converting panics into a stored
/// message and a `0` (failure) return.
///
/// # Safety
/// `user` must be the `ParserInner<H>` registered with the parser, and no
/// other reference to it may be live (true during `cuvidParseVideoData`).
unsafe fn with_handler<H: ParserHandler>(
    user: *mut c_void,
    f: impl FnOnce(&mut H) -> c_int,
) -> c_int {
    if user.is_null() {
        return 0;
    }
    // SAFETY: the caller guarantees exclusivity and validity.
    let inner = unsafe { &mut *(user as *mut ParserInner<H>) };
    if inner.panic.is_some() {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| f(&mut inner.handler))) {
        Ok(v) => v,
        Err(p) => {
            inner.panic = Some(panic_message(p));
            0
        }
    }
}

unsafe extern "system" fn sequence_cb<H: ParserHandler>(
    user: *mut c_void,
    fmt: *mut sys::CUVIDEOFORMAT,
) -> c_int {
    if fmt.is_null() {
        return 0;
    }
    // SAFETY: called by the parser during `parse` with our user pointer and
    // a valid format struct.
    unsafe { with_handler::<H>(user, |h| h.sequence(&*fmt)) }
}

unsafe extern "system" fn decode_cb<H: ParserHandler>(
    user: *mut c_void,
    pic: *mut sys::CUVIDPICPARAMS,
) -> c_int {
    if pic.is_null() {
        return 0;
    }
    // SAFETY: called by the parser with a valid, exclusive picture struct
    // that lives until the callback returns.
    unsafe { with_handler::<H>(user, |h| h.decode(&mut PictureParams { raw: &mut *pic })) }
}

unsafe extern "system" fn display_cb<H: ParserHandler>(
    user: *mut c_void,
    info: *mut sys::CUVIDPARSERDISPINFO,
) -> c_int {
    // SAFETY: called by the parser; `info` is null (EOS) or valid.
    unsafe { with_handler::<H>(user, |h| h.display(info.as_ref())) }
}

/// Packet flags for [`VideoParser::parse`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PacketFlags {
    /// The packet holds exactly one complete picture (lets the parser
    /// decode it immediately instead of waiting for the next start code).
    pub end_of_picture: bool,
    /// Signal a discontinuity before this packet.
    pub discontinuity: bool,
}

impl<H: ParserHandler> VideoParser<H> {
    /// Creates a parser for `codec`.
    ///
    /// `max_display_delay` = 0 gives the lowest latency; `clock_rate` is the
    /// timestamp unit in Hz (only used by the parser for interpolation).
    pub fn new(
        codec: cudaVideoCodec,
        max_display_delay: u32,
        clock_rate: u32,
        handler: H,
    ) -> Result<Self> {
        let api = api()?;
        let inner = Box::into_raw(Box::new(ParserInner { handler, panic: None }));
        let mut params = sys::CUVIDPARSERPARAMS {
            CodecType: codec,
            ulMaxNumDecodeSurfaces: 1,
            ulClockRate: clock_rate,
            ulErrorThreshold: 100,
            ulMaxDisplayDelay: max_display_delay,
            pUserData: inner as *mut c_void,
            pfnSequenceCallback: Some(sequence_cb::<H>),
            pfnDecodePicture: Some(decode_cb::<H>),
            pfnDisplayPicture: Some(display_cb::<H>),
            ..Default::default()
        };
        let mut raw: sys::CUvideoparser = std::ptr::null_mut();
        // SAFETY: `params` is fully initialised; the user pointer stays
        // valid until the parser is destroyed in `Drop`.
        let r = unsafe { (api.cuvidCreateVideoParser)(&mut raw, &mut params) };
        if r != 0 {
            // SAFETY: `inner` came from Box::into_raw above and was never
            // handed to a live parser.
            drop(unsafe { Box::from_raw(inner) });
            return Err(error(r, "cuvidCreateVideoParser"));
        }
        Ok(Self { raw, inner })
    }

    /// Feeds one packet. Callbacks run synchronously inside this call.
    pub fn parse(&mut self, data: &[u8], timestamp: Option<i64>, flags: PacketFlags) -> Result<()> {
        let mut f: c_ulong = 0;
        if timestamp.is_some() {
            f |= sys::CUVID_PKT_TIMESTAMP;
        }
        if flags.end_of_picture {
            f |= sys::CUVID_PKT_ENDOFPICTURE;
        }
        if flags.discontinuity {
            f |= sys::CUVID_PKT_DISCONTINUITY;
        }
        let size = c_ulong::try_from(data.len())
            .map_err(|_| Error::decode("packet too large for NVDEC"))?;
        let mut pkt = sys::CUVIDSOURCEDATAPACKET {
            flags: f,
            payload_size: size,
            payload: data.as_ptr(),
            timestamp: timestamp.unwrap_or(0),
        };
        self.feed(&mut pkt)
    }

    /// Signals end of stream: the parser decodes and displays every
    /// remaining picture. Create a new parser to continue afterwards.
    pub fn end_of_stream(&mut self) -> Result<()> {
        let mut pkt = sys::CUVIDSOURCEDATAPACKET {
            flags: sys::CUVID_PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: std::ptr::null(),
            timestamp: 0,
        };
        self.feed(&mut pkt)
    }

    fn feed(&mut self, pkt: &mut sys::CUVIDSOURCEDATAPACKET) -> Result<()> {
        let api = api()?;
        // SAFETY: valid parser and packet; `&mut self` guarantees no other
        // access to `inner` while callbacks run.
        let r = unsafe { (api.cuvidParseVideoData)(self.raw, pkt) };
        // SAFETY: the parse call has returned, so no callback aliases
        // `inner` any more.
        let inner = unsafe { &mut *self.inner };
        if let Some(msg) = inner.panic.take() {
            return Err(Error::decode(format!("panic in NVDEC callback: {msg}")));
        }
        check(r, "cuvidParseVideoData")
    }

    /// The handler.
    pub fn handler(&self) -> &H {
        // SAFETY: no callback can run while we hold `&self` (parsing needs
        // `&mut self`).
        unsafe { &(*self.inner).handler }
    }

    /// The handler, mutably.
    pub fn handler_mut(&mut self) -> &mut H {
        // SAFETY: as above, with exclusive access.
        unsafe { &mut (*self.inner).handler }
    }

    /// Destroys the parser and returns the handler.
    pub fn into_handler(self) -> H {
        let this = std::mem::ManuallyDrop::new(self);
        this.destroy_parser();
        // SAFETY: the parser is destroyed, so we are the sole owner of the
        // box allocated in `new`; `Drop` will not run (ManuallyDrop).
        let inner = unsafe { Box::from_raw(this.inner) };
        inner.handler
    }

    fn destroy_parser(&self) {
        if let Ok(api) = api() {
            // SAFETY: created in `new`, destroyed once (Drop and
            // into_handler are mutually exclusive).
            unsafe { (api.cuvidDestroyVideoParser)(self.raw) };
        }
    }
}

impl<H: ParserHandler> Drop for VideoParser<H> {
    fn drop(&mut self) {
        self.destroy_parser();
        // SAFETY: the parser no longer references `inner`; the box was
        // created in `new` and is freed exactly once.
        drop(unsafe { Box::from_raw(self.inner) });
    }
}
