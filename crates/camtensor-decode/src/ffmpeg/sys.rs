//! Hand-written FFI for the small part of libavcodec / libavutil we use,
//! loaded at runtime with `libloading`.
//!
//! # ABI surface
//!
//! Only *functions* plus the leading fields of two structs are used. The
//! layouts below were checked against the FFmpeg 8.1 headers
//! (`libavutil/frame.h`, `libavcodec/packet.h`) and the history of those
//! headers for FFmpeg 5.0 – 7.1:
//!
//! * `AVFrame`: `data[8]`, `linesize[8]`, `extended_data`, `width`, `height`,
//!   `nb_samples`, `format` have been the first fields, in this order, since
//!   FFmpeg 2.x. They are followed by `key_frame` (removed in libavutil 60 /
//!   FFmpeg 8.0, `FF_API_FRAME_KEY`), `pict_type`, `sample_aspect_ratio` and
//!   `pts`. `pkt_pts`, which used to sit after `pts`, was removed in
//!   libavutil 57 (FFmpeg 5.0) so it does not matter here. Because `pts` is an
//!   `int64_t` it is 8-byte aligned and lands at offset 136 on 64-bit targets
//!   both with and without `key_frame`; we still model both layouts with
//!   `#[repr(C)]` structs ([`AVFrameTs57`], [`AVFrameTs60`]) and pick one from
//!   the runtime libavutil major version, so 32-bit targets are right too.
//!   Fields after `pts` (colour, crop, buffers) moved several times between
//!   5.x and 8.x and are **not** read; colour information is queried from
//!   the codec context through the AVOption API instead.
//! * `AVPacket`: `buf`, `pts`, `dts`, `data`, `size`, `stream_index`, `flags`
//!   have been the leading fields since FFmpeg 3.x (`FF_API_*` removals only
//!   touched trailing fields such as `convergence_duration`).
//!
//! `AVCodecContext` is treated as opaque: options are set and read with
//! `av_opt_set*` / `av_opt_get_int`, whose option *names* are part of the
//! stable public API.
//!
//! Enum values (`AVCodecID`, `AVPixelFormat`) are resolved by name at load
//! time (`avcodec_find_decoder_by_name`, `av_get_pix_fmt`) so a renumbering
//! in a future major version cannot silently break the mapping. For
//! reference, the FFmpeg 5–8 values are: `AV_CODEC_ID_MJPEG = 7`,
//! `AV_CODEC_ID_H264 = 27`, `AV_CODEC_ID_HEVC = 173` (it shifts once
//! `FF_API_V408_CODECID` is dropped in libavcodec 63) and `yuv420p = 0`,
//! `yuv422p = 4`, `yuv444p = 5`, `gray = 8`, `yuvj420p = 12`, `yuvj422p = 13`,
//! `yuvj444p = 14`, `nv12 = 23`.

#![allow(non_camel_case_types)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use camtensor_core::{Error, Result};
use libloading::Library;

/// Opaque `AVCodec`.
#[repr(C)]
pub struct AVCodec {
    _p: [u8; 0],
}
/// Opaque `AVCodecContext`.
#[repr(C)]
pub struct AVCodecContext {
    _p: [u8; 0],
}
/// Opaque `AVFrame` (see [`AVFrameHead`] for the fields we read).
#[repr(C)]
pub struct AVFrame {
    _p: [u8; 0],
}

/// Number of data pointers in `AVFrame` (`AV_NUM_DATA_POINTERS`).
pub const AV_NUM_DATA_POINTERS: usize = 8;

/// Leading, version-stable fields of `AVFrame`.
#[repr(C)]
pub struct AVFrameHead {
    pub data: [*mut u8; AV_NUM_DATA_POINTERS],
    pub linesize: [c_int; AV_NUM_DATA_POINTERS],
    pub extended_data: *mut *mut u8,
    pub width: c_int,
    pub height: c_int,
    pub nb_samples: c_int,
    pub format: c_int,
}

/// `AVFrame` prefix up to `pts` for libavutil 57–59 (FFmpeg 5.x–7.x).
#[repr(C)]
pub struct AVFrameTs57 {
    pub head: AVFrameHead,
    pub key_frame: c_int,
    pub pict_type: c_int,
    pub sample_aspect_ratio: [c_int; 2],
    pub pts: i64,
}

/// `AVFrame` prefix up to `pts` for libavutil 60+ (FFmpeg 8.x).
#[repr(C)]
pub struct AVFrameTs60 {
    pub head: AVFrameHead,
    pub pict_type: c_int,
    pub sample_aspect_ratio: [c_int; 2],
    pub pts: i64,
}

/// Leading, version-stable fields of `AVPacket`.
#[repr(C)]
pub struct AVPacket {
    pub buf: *mut c_void,
    pub pts: i64,
    pub dts: i64,
    pub data: *mut u8,
    pub size: c_int,
    pub stream_index: c_int,
    pub flags: c_int,
    // more fields follow; never constructed on the Rust side.
}

/// `AV_PKT_FLAG_KEY`.
pub const AV_PKT_FLAG_KEY: c_int = 1;
/// `AV_NOPTS_VALUE`.
pub const AV_NOPTS_VALUE: i64 = i64::MIN;
/// `AV_LOG_QUIET`.
pub const AV_LOG_QUIET: c_int = -8;
/// `AV_LOG_INFO` (libavutil's default level).
pub const AV_LOG_INFO: c_int = 32;

/// `AVCOL_RANGE_JPEG`.
pub const AVCOL_RANGE_JPEG: i64 = 2;
/// `AVCOL_SPC_BT709`.
pub const AVCOL_SPC_BT709: i64 = 1;
/// `AVCOL_SPC_BT470BG`.
pub const AVCOL_SPC_BT470BG: i64 = 5;
/// `AVCOL_SPC_SMPTE170M`.
pub const AVCOL_SPC_SMPTE170M: i64 = 6;

/// `FFERRTAG(a, b, c, d)`.
pub const fn fferrtag(a: u8, b: u8, c: u8, d: u8) -> c_int {
    -((a as u32 | (b as u32) << 8 | (c as u32) << 16 | (d as u32) << 24) as c_int)
}

/// `AVERROR_EOF`.
pub const AVERROR_EOF: c_int = fferrtag(b'E', b'O', b'F', b' ');
/// `AVERROR_INVALIDDATA`.
pub const AVERROR_INVALIDDATA: c_int = fferrtag(b'I', b'N', b'D', b'A');
/// `AVERROR(EAGAIN)`. `EAGAIN` is 11 on Linux and in the MSVC / MinGW CRT,
/// 35 on macOS.
#[cfg(target_os = "macos")]
pub const AVERROR_EAGAIN: c_int = -35;
/// `AVERROR(EAGAIN)`. `EAGAIN` is 11 on Linux and in the MSVC / MinGW CRT,
/// 35 on macOS.
#[cfg(not(target_os = "macos"))]
pub const AVERROR_EAGAIN: c_int = -11;

/// Supported libavcodec major versions (FFmpeg 8.x down to 5.x). The
/// matching libavutil major is `avcodec_major - 2`.
pub const AVCODEC_MAJORS: [u32; 4] = [62, 61, 60, 59];

/// Pixel-format ids resolved by name (`-1` when unknown to the library).
#[derive(Debug, Clone, Copy)]
pub struct PixFmtIds {
    pub yuv420p: c_int,
    pub yuvj420p: c_int,
    pub yuv422p: c_int,
    pub yuvj422p: c_int,
    pub yuv444p: c_int,
    pub yuvj444p: c_int,
    pub nv12: c_int,
    pub gray: c_int,
}

/// Function table of the loaded libraries.
pub struct Api {
    _avutil: Library,
    _avcodec: Library,
    /// Where the libraries were found (for diagnostics).
    pub location: String,
    pub avcodec_major: u32,
    pub avutil_major: u32,
    pub pix: PixFmtIds,

    pub avcodec_find_decoder: unsafe extern "C" fn(c_int) -> *const AVCodec,
    pub avcodec_find_decoder_by_name: unsafe extern "C" fn(*const c_char) -> *const AVCodec,
    pub avcodec_alloc_context3: unsafe extern "C" fn(*const AVCodec) -> *mut AVCodecContext,
    pub avcodec_free_context: unsafe extern "C" fn(*mut *mut AVCodecContext),
    pub avcodec_open2:
        unsafe extern "C" fn(*mut AVCodecContext, *const AVCodec, *mut *mut c_void) -> c_int,
    pub avcodec_send_packet: unsafe extern "C" fn(*mut AVCodecContext, *const AVPacket) -> c_int,
    pub avcodec_receive_frame: unsafe extern "C" fn(*mut AVCodecContext, *mut AVFrame) -> c_int,
    pub avcodec_flush_buffers: unsafe extern "C" fn(*mut AVCodecContext),
    pub av_packet_alloc: unsafe extern "C" fn() -> *mut AVPacket,
    pub av_packet_free: unsafe extern "C" fn(*mut *mut AVPacket),
    pub av_packet_unref: unsafe extern "C" fn(*mut AVPacket),
    pub av_new_packet: unsafe extern "C" fn(*mut AVPacket, c_int) -> c_int,

    pub av_frame_alloc: unsafe extern "C" fn() -> *mut AVFrame,
    pub av_frame_free: unsafe extern "C" fn(*mut *mut AVFrame),
    pub av_frame_unref: unsafe extern "C" fn(*mut AVFrame),
    pub av_get_pix_fmt_name: unsafe extern "C" fn(c_int) -> *const c_char,
    pub av_strerror: unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int,
    pub av_opt_set_int: unsafe extern "C" fn(*mut c_void, *const c_char, i64, c_int) -> c_int,
    pub av_opt_set: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, c_int) -> c_int,
    pub av_opt_get_int: unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut i64) -> c_int,
}

impl std::fmt::Debug for Api {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Api")
            .field("location", &self.location)
            .field("avcodec_major", &self.avcodec_major)
            .field("avutil_major", &self.avutil_major)
            .finish_non_exhaustive()
    }
}

impl Api {
    /// Human-readable error string for an `AVERROR` code.
    pub fn err_str(&self, code: c_int) -> String {
        let mut buf = [0 as c_char; 128];
        // SAFETY: `buf` is a writable buffer of the advertised size;
        // av_strerror always NUL-terminates within it.
        let r = unsafe { (self.av_strerror)(code, buf.as_mut_ptr(), buf.len()) };
        if r < 0 {
            return format!("error {code}");
        }
        // SAFETY: NUL-terminated by av_strerror (see above).
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) };
        format!("{} ({code})", s.to_string_lossy())
    }

    /// Name of a pixel format, e.g. `yuv420p10le`.
    pub fn pix_fmt_name(&self, fmt: c_int) -> String {
        // SAFETY: av_get_pix_fmt_name accepts any value and returns NULL for
        // unknown ones; the returned string is static.
        let p = unsafe { (self.av_get_pix_fmt_name)(fmt) };
        if p.is_null() {
            return format!("pix_fmt#{fmt}");
        }
        // SAFETY: non-null, static NUL-terminated string owned by libavutil.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

/// Returns the loaded libraries, loading them on first use.
pub fn api() -> Result<&'static Api> {
    API.get_or_init(load).as_ref().map_err(Error::unavailable)
}

/// File names for a given libavcodec major version.
fn lib_names(avcodec_major: u32) -> (String, String) {
    let avutil_major = avcodec_major - 2;
    if cfg!(windows) {
        (format!("avutil-{avutil_major}.dll"), format!("avcodec-{avcodec_major}.dll"))
    } else if cfg!(target_os = "macos") {
        (format!("libavutil.{avutil_major}.dylib"), format!("libavcodec.{avcodec_major}.dylib"))
    } else {
        (format!("libavutil.so.{avutil_major}"), format!("libavcodec.so.{avcodec_major}"))
    }
}

/// Directories searched before the system loader.
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("CAMTENSOR_FFMPEG_DIR").filter(|d| !d.is_empty()) {
        let d = PathBuf::from(d);
        // Accept both `<ffmpeg>/bin` (Windows builds) / `<ffmpeg>/lib` and
        // the root of an extracted FFmpeg build.
        dirs.push(d.join(if cfg!(windows) { "bin" } else { "lib" }));
        dirs.insert(0, d);
    }
    if let Some(d) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(d);
    }
    dirs
}

fn open_lib(path: &Path) -> std::result::Result<Library, libloading::Error> {
    #[cfg(windows)]
    if path.is_absolute() {
        use libloading::os::windows::{
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        };
        // SAFETY: loading FFmpeg runs its DllMain/static constructors, which
        // have no preconditions. The DLL-dir flag lets the loader resolve
        // avcodec's own dependencies (avutil, swresample) next to it.
        return unsafe {
            libloading::os::windows::Library::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .map(Library::from);
    }
    // SAFETY: as above, FFmpeg's initialisers have no preconditions.
    unsafe { Library::new(path) }
}

fn load() -> std::result::Result<Api, String> {
    let mut tried = Vec::new();
    let dirs = search_dirs();
    for major in AVCODEC_MAJORS {
        let (util_name, codec_name) = lib_names(major);
        let mut candidates: Vec<(PathBuf, PathBuf)> = dirs
            .iter()
            .map(|d| (d.join(&util_name), d.join(&codec_name)))
            .filter(|(u, c)| u.is_file() && c.is_file())
            .collect();
        // System loader (PATH / LD_LIBRARY_PATH / ldconfig cache).
        candidates.push((PathBuf::from(&util_name), PathBuf::from(&codec_name)));
        for (util_path, codec_path) in candidates {
            match try_load(&util_path, &codec_path, major) {
                Ok(api) => {
                    tracing::debug!(
                        location = %api.location,
                        avcodec = api.avcodec_major,
                        avutil = api.avutil_major,
                        "loaded libavcodec"
                    );
                    return Ok(api);
                }
                Err(e) => tried.push(format!("{}: {e}", codec_path.display())),
            }
        }
    }
    Err(format!(
        "libavcodec (FFmpeg 5.x–8.x shared libraries) not found. Install them \
         (Debian/Ubuntu: `apt install libavcodec-dev` or the runtime package `libavcodec6x`; \
         Fedora: `dnf install ffmpeg-libs`; Windows: download a *shared* LGPL build such as \
         ffmpeg-n8.1-latest-win64-lgpl-shared) and either put the libraries next to the \
         executable, on the library path, or set CAMTENSOR_FFMPEG_DIR to the directory \
         containing {}. Searched {} then the system loader; attempts: {}",
        lib_names(AVCODEC_MAJORS[0]).1,
        if dirs.is_empty() {
            "no directory".to_string()
        } else {
            dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
        },
        tried.join("; ")
    ))
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {{
        // SAFETY: the symbol is looked up by its exact C name and assigned
        // to a field whose type matches the C prototype of that function in
        // FFmpeg 5.x–8.x (checked against the 8.1 headers).
        let s = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .map_err(|e| format!("missing symbol {}: {e}", $name))?;
        *s
    }};
}

fn try_load(util_path: &Path, codec_path: &Path, major: u32) -> std::result::Result<Api, String> {
    // libavutil first so that libavcodec's dependency resolves to the very
    // same library (same soname / module already loaded).
    let avutil = open_lib(util_path).map_err(|e| e.to_string())?;
    let avcodec = open_lib(codec_path).map_err(|e| e.to_string())?;

    let avutil_version: unsafe extern "C" fn() -> c_uint = sym!(avutil, "avutil_version");
    let avcodec_version: unsafe extern "C" fn() -> c_uint = sym!(avcodec, "avcodec_version");
    // SAFETY: version getters take no arguments and have no preconditions.
    let (uv, cv) = unsafe { (avutil_version(), avcodec_version()) };
    let (avutil_major, avcodec_major) = (uv >> 16, cv >> 16);
    if avcodec_major != major || avutil_major != major - 2 {
        return Err(format!(
            "version mismatch: libavcodec {avcodec_major} / libavutil {avutil_major}"
        ));
    }

    let av_get_pix_fmt: unsafe extern "C" fn(*const c_char) -> c_int =
        sym!(avutil, "av_get_pix_fmt");
    let pf = |name: &CStr| {
        // SAFETY: NUL-terminated name; returns AV_PIX_FMT_NONE (-1) if unknown.
        unsafe { av_get_pix_fmt(name.as_ptr()) }
    };
    let pix = PixFmtIds {
        yuv420p: pf(c"yuv420p"),
        yuvj420p: pf(c"yuvj420p"),
        yuv422p: pf(c"yuv422p"),
        yuvj422p: pf(c"yuvj422p"),
        yuv444p: pf(c"yuv444p"),
        yuvj444p: pf(c"yuvj444p"),
        nv12: pf(c"nv12"),
        gray: pf(c"gray"),
    };

    // Keep libavcodec quiet by default: it logs every concealed slice to
    // stderr, and some per-packet errors at AV_LOG_FATAL (e.g. mjpeg's "No
    // JPEG data found"); camtensor reports them through `tracing`. Respect a level the host application already changed and
    // allow an explicit override.
    let av_log_set_level: unsafe extern "C" fn(c_int) = sym!(avutil, "av_log_set_level");
    let av_log_get_level: unsafe extern "C" fn() -> c_int = sym!(avutil, "av_log_get_level");
    let wanted = std::env::var("CAMTENSOR_FFMPEG_LOG").ok().and_then(|v| parse_log_level(&v));
    // SAFETY: plain global setters/getters without preconditions.
    unsafe {
        match wanted {
            Some(level) => av_log_set_level(level),
            None if av_log_get_level() == AV_LOG_INFO => av_log_set_level(AV_LOG_QUIET),
            None => {}
        }
    }

    let location = codec_path.display().to_string();
    Ok(Api {
        avcodec_find_decoder: sym!(avcodec, "avcodec_find_decoder"),
        avcodec_find_decoder_by_name: sym!(avcodec, "avcodec_find_decoder_by_name"),
        avcodec_alloc_context3: sym!(avcodec, "avcodec_alloc_context3"),
        avcodec_free_context: sym!(avcodec, "avcodec_free_context"),
        avcodec_open2: sym!(avcodec, "avcodec_open2"),
        avcodec_send_packet: sym!(avcodec, "avcodec_send_packet"),
        avcodec_receive_frame: sym!(avcodec, "avcodec_receive_frame"),
        avcodec_flush_buffers: sym!(avcodec, "avcodec_flush_buffers"),
        av_packet_alloc: sym!(avcodec, "av_packet_alloc"),
        av_packet_free: sym!(avcodec, "av_packet_free"),
        av_packet_unref: sym!(avcodec, "av_packet_unref"),
        av_new_packet: sym!(avcodec, "av_new_packet"),
        av_frame_alloc: sym!(avutil, "av_frame_alloc"),
        av_frame_free: sym!(avutil, "av_frame_free"),
        av_frame_unref: sym!(avutil, "av_frame_unref"),
        av_get_pix_fmt_name: sym!(avutil, "av_get_pix_fmt_name"),
        av_strerror: sym!(avutil, "av_strerror"),
        av_opt_set_int: sym!(avutil, "av_opt_set_int"),
        av_opt_set: sym!(avutil, "av_opt_set"),
        av_opt_get_int: sym!(avutil, "av_opt_get_int"),
        _avutil: avutil,
        _avcodec: avcodec,
        location,
        avcodec_major,
        avutil_major,
        pix,
    })
}

/// Parses `CAMTENSOR_FFMPEG_LOG` (name or number).
fn parse_log_level(v: &str) -> Option<c_int> {
    Some(match v.trim().to_ascii_lowercase().as_str() {
        "quiet" => -8,
        "panic" => 0,
        "fatal" => 8,
        "error" => 16,
        "warning" | "warn" => 24,
        "info" => 32,
        "verbose" => 40,
        "debug" => 48,
        "trace" => 56,
        other => other.parse().ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_tags_match_headers() {
        // Values computed by FFERRTAG in libavutil/error.h.
        assert_eq!(AVERROR_EOF, -541_478_725);
        assert_eq!(AVERROR_INVALIDDATA, -1_094_995_529);
    }

    #[test]
    fn frame_pts_offset_is_stable() {
        // On 64-bit targets `pts` sits at offset 136 with and without the
        // removed `key_frame` field (alignment padding absorbs it).
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::offset_of!(AVFrameTs57, pts), 136);
            assert_eq!(std::mem::offset_of!(AVFrameTs60, pts), 136);
            assert_eq!(std::mem::offset_of!(AVFrameHead, format), 116);
            assert_eq!(std::mem::offset_of!(AVPacket, size), 32);
        }
    }

    #[test]
    fn log_levels() {
        assert_eq!(parse_log_level("error"), Some(16));
        assert_eq!(parse_log_level("24"), Some(24));
        assert_eq!(parse_log_level("nope"), None);
    }
}
