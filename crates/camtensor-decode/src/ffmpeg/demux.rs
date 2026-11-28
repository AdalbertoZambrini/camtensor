//! Video-file demuxing through a runtime-loaded libavformat (FFmpeg 6.x–8.x).
//!
//! Only the video stream is read. Packets come out exactly as stored in
//! the container: length-prefixed (AVCC/HVCC) for MP4/MKV, Annex B for raw
//! `.h264`/`.hevc` and MPEG-TS, complete JPEG images for MJPEG. The
//! container's codec configuration record is returned as `extradata`.
//!
//! # ABI notes
//!
//! Only stable leading fields are read, checked against the headers of
//! FFmpeg 6.0, 6.1, 7.0, 7.1 and 8.0:
//! * `AVFormatContext`: `av_class, iformat, oformat, priv_data, pb,
//!   ctx_flags, nb_streams, streams`;
//! * `AVStream` (libavformat ≥ 60): `av_class, index, id, codecpar,
//!   priv_data, time_base, start_time, duration, nb_frames, disposition,
//!   discard, sample_aspect_ratio, metadata, avg_frame_rate`;
//! * `AVCodecParameters`: `codec_type, codec_id, codec_tag, extradata,
//!   extradata_size`.
//!
//! libavformat 59 (FFmpeg 5.x) stores `codecpar` later in `AVStream`, so it
//! is rejected for file input (RTSP input does not use libavformat).
//!
//! Capture devices (webcams) are opened through libavdevice (`dshow`,
//! `v4l2`, `avfoundation`). For uncompressed video the size and pixel
//! format are read from `AVCodecParameters`, whose layout differs between
//! libavformat 60.0–60.15 (FFmpeg 6.0) and 60.16+ (FFmpeg 6.1+, which
//! inserted `coded_side_data`); both layouts are handled.

use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use libloading::Library;

use camtensor_core::{Codec, Error, Result};

use super::sys::{self, AV_NOPTS_VALUE, AV_PKT_FLAG_KEY, AVPacket};

#[repr(C)]
struct AVRational {
    num: c_int,
    den: c_int,
}

#[repr(C)]
struct AVFormatContextHead {
    av_class: *const c_void,
    iformat: *const c_void,
    oformat: *const c_void,
    priv_data: *mut c_void,
    pb: *mut c_void,
    ctx_flags: c_int,
    nb_streams: c_uint,
    streams: *mut *mut AVStreamHead,
}

#[repr(C)]
struct AVStreamHead {
    av_class: *const c_void,
    index: c_int,
    id: c_int,
    codecpar: *mut AVCodecParametersHead,
    priv_data: *mut c_void,
    time_base: AVRational,
    start_time: i64,
    duration: i64,
    nb_frames: i64,
    disposition: c_int,
    discard: c_int,
    sample_aspect_ratio: AVRational,
    metadata: *mut c_void,
    avg_frame_rate: AVRational,
}

#[repr(C)]
struct AVCodecParametersHead {
    codec_type: c_int,
    codec_id: c_int,
    codec_tag: u32,
    extradata: *mut u8,
    extradata_size: c_int,
}

/// `AVCodecParameters` up to `height`, FFmpeg 6.1+ (libavformat >= 60.16).
#[repr(C)]
struct CodecParNew {
    codec_type: c_int,
    codec_id: c_int,
    codec_tag: u32,
    extradata: *mut u8,
    extradata_size: c_int,
    coded_side_data: *mut c_void,
    nb_coded_side_data: c_int,
    format: c_int,
    bit_rate: i64,
    bits_per_coded_sample: c_int,
    bits_per_raw_sample: c_int,
    profile: c_int,
    level: c_int,
    width: c_int,
    height: c_int,
}

/// `AVCodecParameters` up to `height`, FFmpeg 6.0 (libavformat 60.0–60.15).
#[repr(C)]
struct CodecParOld {
    codec_type: c_int,
    codec_id: c_int,
    codec_tag: u32,
    extradata: *mut u8,
    extradata_size: c_int,
    format: c_int,
    bit_rate: i64,
    bits_per_coded_sample: c_int,
    bits_per_raw_sample: c_int,
    profile: c_int,
    level: c_int,
    width: c_int,
    height: c_int,
}

#[repr(C)]
struct AVDeviceInfo {
    device_name: *mut c_char,
    device_description: *mut c_char,
    media_types: *mut c_int,
    nb_media_types: c_int,
}

#[repr(C)]
struct AVDeviceInfoList {
    devices: *mut *mut AVDeviceInfo,
    nb_devices: c_int,
    default_device: c_int,
}

type FmtCtx = c_void;

struct FormatApi {
    _lib: Library,
    minor: u32,
    av_find_input_format: unsafe extern "C" fn(*const c_char) -> *const c_void,
    avformat_open_input: unsafe extern "C" fn(
        *mut *mut FmtCtx,
        *const c_char,
        *const c_void,
        *mut *mut c_void,
    ) -> c_int,
    avformat_find_stream_info: unsafe extern "C" fn(*mut FmtCtx, *mut *mut c_void) -> c_int,
    av_find_best_stream:
        unsafe extern "C" fn(*mut FmtCtx, c_int, c_int, c_int, *mut *const c_void, c_int) -> c_int,
    av_read_frame: unsafe extern "C" fn(*mut FmtCtx, *mut AVPacket) -> c_int,
    av_seek_frame: unsafe extern "C" fn(*mut FmtCtx, c_int, i64, c_int) -> c_int,
    avformat_close_input: unsafe extern "C" fn(*mut *mut FmtCtx),
}

static FORMAT: OnceLock<std::result::Result<FormatApi, String>> = OnceLock::new();

fn format_api() -> Result<&'static FormatApi> {
    FORMAT.get_or_init(load_format).as_ref().map_err(Error::unavailable)
}

fn load_format() -> std::result::Result<FormatApi, String> {
    let api = sys::api().map_err(|e| e.to_string())?;
    let major = api.avcodec_major;
    if major < 60 {
        return Err(format!(
            "file input needs FFmpeg 6 or newer (libavformat >= 60); found libavcodec {major}"
        ));
    }
    let lib = open_sibling(&api.location, "avformat", major)?;
    bind(lib)
}

/// File name of an FFmpeg library for this platform.
fn lib_file(base: &str, major: u32) -> String {
    if cfg!(windows) {
        format!("{base}-{major}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{base}.{major}.dylib")
    } else {
        format!("lib{base}.so.{major}")
    }
}

/// Opens `base` from the directory libavcodec was loaded from, then from
/// the system loader.
fn open_sibling(
    avcodec_location: &str,
    base: &str,
    major: u32,
) -> std::result::Result<Library, String> {
    let name = lib_file(base, major);
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = Path::new(avcodec_location).parent().filter(|d| d.is_dir()) {
        candidates.push(dir.join(&name));
    }
    candidates.push(PathBuf::from(&name));
    let mut errors = Vec::new();
    for c in candidates {
        // Same loader as libavcodec: on Windows, dependencies (avfilter,
        // swscale, ...) resolve from the library's own directory.
        match sys::open_lib(&c) {
            Ok(lib) => return Ok(lib),
            Err(e) => errors.push(format!("{}: {e}", c.display())),
        }
    }
    Err(format!("lib{base} not found ({})", errors.join("; ")))
}

/// libavdevice and the libavutil dictionary helpers (capture devices).
struct DeviceApi {
    _dev: Library,
    _util: Library,
    avdevice_list_input_sources: unsafe extern "C" fn(
        *const c_void,
        *const c_char,
        *mut c_void,
        *mut *mut AVDeviceInfoList,
    ) -> c_int,
    avdevice_free_list_devices: unsafe extern "C" fn(*mut *mut AVDeviceInfoList),
    av_dict_set:
        unsafe extern "C" fn(*mut *mut c_void, *const c_char, *const c_char, c_int) -> c_int,
    av_dict_free: unsafe extern "C" fn(*mut *mut c_void),
}

static DEVICE: OnceLock<std::result::Result<DeviceApi, String>> = OnceLock::new();

fn device_api() -> Result<&'static DeviceApi> {
    DEVICE.get_or_init(load_device).as_ref().map_err(Error::unavailable)
}

fn load_device() -> std::result::Result<DeviceApi, String> {
    format_api().map_err(|e| e.to_string())?;
    let api = sys::api().map_err(|e| e.to_string())?;
    let major = api.avcodec_major;
    let dev = open_sibling(&api.location, "avdevice", major)?;
    let util = open_sibling(&api.location, "avutil", major - 2)?;
    macro_rules! sym {
        ($lib:ident, $name:literal) => {{
            // SAFETY: looked up by C name; field types match the FFmpeg 6–8
            // prototypes.
            let s = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
                .map_err(|e| format!("missing symbol {}: {e}", $name))?;
            *s
        }};
    }
    let register: unsafe extern "C" fn() = sym!(dev, "avdevice_register_all");
    // SAFETY: registers the device (de)muxers; idempotent, no preconditions.
    unsafe { register() };
    Ok(DeviceApi {
        avdevice_list_input_sources: sym!(dev, "avdevice_list_input_sources"),
        avdevice_free_list_devices: sym!(dev, "avdevice_free_list_devices"),
        av_dict_set: sym!(util, "av_dict_set"),
        av_dict_free: sym!(util, "av_dict_free"),
        _dev: dev,
        _util: util,
    })
}

/// Default capture backend of this platform.
pub fn default_capture_backend() -> &'static str {
    if cfg!(windows) {
        "dshow"
    } else if cfg!(target_os = "macos") {
        "avfoundation"
    } else {
        "v4l2"
    }
}

/// Lists capture devices of `backend` as `(name, description)`.
pub fn list_devices(backend: &str) -> Result<Vec<(String, String)>> {
    let fmt = format_api()?;
    let dev = device_api()?;
    let b = CString::new(backend).map_err(|_| Error::config("bad backend name"))?;
    // SAFETY: NUL-terminated name; returns NULL if unknown.
    let ifmt = unsafe { (fmt.av_find_input_format)(b.as_ptr()) };
    if ifmt.is_null() {
        return Err(Error::unavailable(format!("capture backend '{backend}' not available")));
    }
    let mut list: *mut AVDeviceInfoList = std::ptr::null_mut();
    // SAFETY: valid input format, no device name/options, valid out-pointer.
    let r = unsafe {
        (dev.avdevice_list_input_sources)(ifmt, std::ptr::null(), std::ptr::null_mut(), &mut list)
    };
    if r < 0 || list.is_null() {
        return Err(Error::Ingest(format!("cannot list {backend} devices ({r})")));
    }
    let mut out = Vec::new();
    // SAFETY: `list` was filled by libavdevice; strings are NUL-terminated;
    // the list is freed exactly once below.
    unsafe {
        let l = &*list;
        for i in 0..l.nb_devices.max(0) as usize {
            let d = &**l.devices.add(i);
            let s = |p: *mut c_char| {
                if p.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            // Keep video-capable devices (AVMEDIA_TYPE_VIDEO = 0); devices
            // without media-type information are kept too.
            let types = if d.media_types.is_null() || d.nb_media_types <= 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(d.media_types, d.nb_media_types as usize)
            };
            if !types.is_empty() && !types.contains(&0) {
                continue;
            }
            out.push((s(d.device_name), s(d.device_description)));
        }
        (dev.avdevice_free_list_devices)(&mut list);
    }
    Ok(out)
}

fn bind(lib: Library) -> std::result::Result<FormatApi, String> {
    macro_rules! sym {
        ($name:literal) => {{
            // SAFETY: the symbol is looked up by its C name and stored in a
            // field whose type matches its prototype in FFmpeg 6.x–8.x.
            let s = unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                .map_err(|e| format!("missing symbol {}: {e}", $name))?;
            *s
        }};
    }
    let version: unsafe extern "C" fn() -> c_uint = sym!("avformat_version");
    // SAFETY: no arguments, no preconditions.
    let major = unsafe { version() } >> 16;
    if major < 60 {
        return Err(format!("libavformat {major} is too old (need >= 60)"));
    }
    // SAFETY: no arguments, no preconditions.
    let minor = (unsafe { version() } >> 8) & 0xff;
    Ok(FormatApi {
        minor,
        av_find_input_format: sym!("av_find_input_format"),
        avformat_open_input: sym!("avformat_open_input"),
        avformat_find_stream_info: sym!("avformat_find_stream_info"),
        av_find_best_stream: sym!("av_find_best_stream"),
        av_read_frame: sym!("av_read_frame"),
        av_seek_frame: sym!("av_seek_frame"),
        avformat_close_input: sym!("avformat_close_input"),
        _lib: lib,
    })
}

/// Static description of the video stream of a file.
#[derive(Debug, Clone)]
pub struct DemuxInfo {
    /// Codec of the video stream.
    pub codec: Codec,
    /// Codec configuration record (avcC / hvcC) or Annex B parameter sets
    /// as stored by the container; may be empty.
    pub extradata: Vec<u8>,
    /// Average frame rate if known.
    pub fps: Option<f64>,
    /// Size and FFmpeg pixel format id of uncompressed video.
    pub raw: Option<RawVideoInfo>,
}

/// Uncompressed video description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawVideoInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `AVPixelFormat` id (e.g. 1 = YUYV422, 15 = UYVY422, 23 = NV12).
    pub pix_fmt: i32,
}

/// One demuxed video packet.
#[derive(Debug, Clone)]
pub struct DemuxPacket {
    /// Payload as stored in the container.
    pub data: Vec<u8>,
    /// Presentation time in seconds from the start of the file (`None` if
    /// the container has no timestamp for this packet).
    pub pts_seconds: Option<f64>,
    /// Container keyframe flag.
    pub keyframe: bool,
}

/// Reads the video stream of a media file.
pub struct Demuxer {
    fmt: &'static FormatApi,
    api: &'static sys::Api,
    ctx: *mut FmtCtx,
    pkt: *mut AVPacket,
    index: c_int,
    time_base: (i32, i32),
    start: i64,
    info: DemuxInfo,
}

// SAFETY: the format context and packet are owned exclusively by this value
// and only used through `&mut self`.
unsafe impl Send for Demuxer {}

impl std::fmt::Debug for Demuxer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Demuxer").field("info", &self.info).finish_non_exhaustive()
    }
}

impl Demuxer {
    /// Opens `path` and selects its best video stream.
    pub fn open(path: &Path) -> Result<Self> {
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error::config("file path contains a NUL byte"))?;
        Self::open_input(&cpath, std::ptr::null(), &[], &path.display().to_string())
    }

    /// Opens a capture device, e.g. `("dshow", "video=USB Camera")` or
    /// `("v4l2", "/dev/video0")`, with demuxer options such as
    /// `video_size`, `framerate`, `pixel_format` / `input_format`.
    pub fn open_device(backend: &str, device: &str, options: &[(String, String)]) -> Result<Self> {
        let fmt = format_api()?;
        device_api()?;
        let b = CString::new(backend).map_err(|_| Error::config("bad backend name"))?;
        // SAFETY: NUL-terminated name; returns NULL if unknown.
        let ifmt = unsafe { (fmt.av_find_input_format)(b.as_ptr()) };
        if ifmt.is_null() {
            return Err(Error::unavailable(format!("capture backend '{backend}' not available")));
        }
        let url =
            CString::new(device).map_err(|_| Error::config("device name contains a NUL byte"))?;
        Self::open_input(&url, ifmt, options, &format!("{backend}:{device}"))
    }

    fn open_input(
        url: &CString,
        ifmt: *const c_void,
        options: &[(String, String)],
        label: &str,
    ) -> Result<Self> {
        let fmt = format_api()?;
        let api = sys::api()?;
        let mut dict: *mut c_void = std::ptr::null_mut();
        if !options.is_empty() {
            let dev = device_api()?;
            for (k, v) in options {
                let (k, v) = (
                    CString::new(k.as_str()).map_err(|_| Error::config("bad option"))?,
                    CString::new(v.as_str()).map_err(|_| Error::config("bad option"))?,
                );
                // SAFETY: valid dictionary out-pointer and NUL-terminated strings
                // (copied by libavutil).
                unsafe { (dev.av_dict_set)(&mut dict, k.as_ptr(), v.as_ptr(), 0) };
            }
        }
        let mut ctx: *mut FmtCtx = std::ptr::null_mut();
        // SAFETY: valid out-pointer, NUL-terminated URL, optional input
        // format and options dictionary (consumed entries are removed).
        let r = unsafe {
            (fmt.avformat_open_input)(
                &mut ctx,
                url.as_ptr(),
                ifmt,
                if dict.is_null() { std::ptr::null_mut() } else { &mut dict },
            )
        };
        if !dict.is_null()
            && let Ok(dev) = device_api()
        {
            // SAFETY: dictionary allocated by av_dict_set.
            unsafe { (dev.av_dict_free)(&mut dict) };
        }
        let path = label;
        if r < 0 {
            return Err(Error::Ingest(format!("cannot open {path}: {}", api.err_str(r))));
        }
        let mut d = Demuxer {
            fmt,
            api,
            ctx,
            pkt: std::ptr::null_mut(),
            index: -1,
            time_base: (1, 90_000),
            start: 0,
            info: DemuxInfo { codec: Codec::H264, extradata: Vec::new(), fps: None, raw: None },
        };
        // SAFETY: `ctx` is an open format context.
        let r = unsafe { (fmt.avformat_find_stream_info)(ctx, std::ptr::null_mut()) };
        if r < 0 {
            return Err(Error::Ingest(format!("cannot read {path}: {}", api.err_str(r))));
        }
        const AVMEDIA_TYPE_VIDEO: c_int = 0;
        // SAFETY: open context; no decoder out-pointer requested.
        let idx = unsafe {
            (fmt.av_find_best_stream)(ctx, AVMEDIA_TYPE_VIDEO, -1, -1, std::ptr::null_mut(), 0)
        };
        if idx < 0 {
            return Err(Error::Ingest(format!("{path}: no video stream")));
        }
        let new_layout = fmt.minor >= 16 || sys::api()?.avcodec_major > 60;
        // SAFETY: the context and its stream array are valid after
        // find_stream_info; `idx` < nb_streams as returned by libavformat;
        // field layouts are documented in the module docs.
        let (codec_id, extradata, time_base, fps, start, raw) = unsafe {
            let head = &*(ctx as *const AVFormatContextHead);
            if idx as c_uint >= head.nb_streams {
                return Err(Error::Ingest("invalid stream index".into()));
            }
            let st = &**head.streams.add(idx as usize);
            let par = &*st.codecpar;
            let extra = if par.extradata.is_null() || par.extradata_size <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(par.extradata, par.extradata_size as usize).to_vec()
            };
            let fps = (st.avg_frame_rate.num > 0 && st.avg_frame_rate.den > 0)
                .then(|| st.avg_frame_rate.num as f64 / st.avg_frame_rate.den as f64);
            let start = if st.start_time == AV_NOPTS_VALUE { 0 } else { st.start_time };
            let (w, h, f) = if new_layout {
                let r = &*(st.codecpar as *const CodecParNew);
                (r.width, r.height, r.format)
            } else {
                let r = &*(st.codecpar as *const CodecParOld);
                (r.width, r.height, r.format)
            };
            let raw = (w > 0 && h > 0).then_some(RawVideoInfo {
                width: w as u32,
                height: h as u32,
                pix_fmt: f,
            });
            (par.codec_id, extra, (st.time_base.num, st.time_base.den), fps, start, raw)
        };
        let codec = match codec_id {
            27 => Codec::H264,
            173 => Codec::H265,
            7 => Codec::Mjpeg,
            13 => Codec::RawVideo,
            other => {
                return Err(Error::Unavailable(format!(
                    "{path}: unsupported video codec id {other} (H.264, H.265, MJPEG and raw video are supported)"
                )));
            }
        };
        // SAFETY: allocation without preconditions.
        d.pkt = unsafe { (api.av_packet_alloc)() };
        if d.pkt.is_null() {
            return Err(Error::decode("av_packet_alloc failed"));
        }
        d.index = idx;
        d.time_base = if time_base.0 > 0 && time_base.1 > 0 { time_base } else { (1, 90_000) };
        d.start = start;
        d.info = DemuxInfo {
            codec,
            extradata,
            fps,
            raw: if codec == Codec::RawVideo { raw } else { None },
        };
        Ok(d)
    }

    /// Stream description.
    pub fn info(&self) -> &DemuxInfo {
        &self.info
    }

    /// Reads the next video packet; `Ok(None)` at the end of the file.
    pub fn read(&mut self) -> Result<Option<DemuxPacket>> {
        loop {
            // SAFETY: open context and allocated packet.
            let r = unsafe { (self.fmt.av_read_frame)(self.ctx, self.pkt) };
            if r < 0 {
                const AVERROR_EOF: c_int = sys::fferrtag(b'E', b'O', b'F', b' ');
                if r == AVERROR_EOF {
                    return Ok(None);
                }
                return Err(Error::Ingest(format!("read error: {}", self.api.err_str(r))));
            }
            // SAFETY: av_read_frame filled the packet; we copy what we need
            // and unref it before the next read.
            let out = unsafe {
                let p = &*self.pkt;
                let keep = p.stream_index == self.index && !p.data.is_null() && p.size > 0;
                let res = keep.then(|| {
                    let data = std::slice::from_raw_parts(p.data, p.size as usize).to_vec();
                    let ts = if p.pts != AV_NOPTS_VALUE { p.pts } else { p.dts };
                    let pts_seconds = (ts != AV_NOPTS_VALUE).then(|| {
                        (ts - self.start) as f64 * self.time_base.0 as f64 / self.time_base.1 as f64
                    });
                    DemuxPacket { data, pts_seconds, keyframe: p.flags & AV_PKT_FLAG_KEY != 0 }
                });
                (self.api.av_packet_unref)(self.pkt);
                res
            };
            if let Some(pkt) = out {
                return Ok(Some(pkt));
            }
        }
    }

    /// Seeks back to the beginning of the file.
    pub fn rewind(&mut self) -> Result<()> {
        const AVSEEK_FLAG_BACKWARD: c_int = 1;
        // SAFETY: open context; timestamp in the stream's time base.
        let r = unsafe {
            (self.fmt.av_seek_frame)(self.ctx, self.index, self.start, AVSEEK_FLAG_BACKWARD)
        };
        if r < 0 {
            return Err(Error::Ingest(format!("seek failed: {}", self.api.err_str(r))));
        }
        Ok(())
    }
}

impl Drop for Demuxer {
    fn drop(&mut self) {
        // SAFETY: both pointers were allocated by libav* (or are null) and
        // are released exactly once here.
        unsafe {
            if !self.pkt.is_null() {
                (self.api.av_packet_free)(&mut self.pkt);
            }
            if !self.ctx.is_null() {
                (self.fmt.avformat_close_input)(&mut self.ctx);
            }
        }
    }
}
