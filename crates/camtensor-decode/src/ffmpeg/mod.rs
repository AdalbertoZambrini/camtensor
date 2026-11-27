//! libavcodec software decoder (H.264, H.265, MJPEG), loaded at runtime.
//!
//! # Library discovery
//!
//! libavcodec and libavutil (FFmpeg 5.x – 8.x, i.e. libavcodec majors
//! 59 – 62) are loaded once per process, searching in order:
//!
//! 1. `$CAMTENSOR_FFMPEG_DIR` (and its `bin/` on Windows, `lib/` elsewhere),
//! 2. the directory of the current executable,
//! 3. the system loader (`PATH` on Windows, `LD_LIBRARY_PATH` / ldconfig on
//!    Linux).
//!
//! Newer majors are preferred. If nothing is found,
//! [`FfmpegDecoder::open`] fails with [`Error::Unavailable`].
//!
//! libavcodec logging is silenced (`quiet`) unless the host application
//! already changed the level; set `CAMTENSOR_FFMPEG_LOG=error|warning|info|…`
//! to override.
//!
//! # Threads and latency
//!
//! `threads = 0` selects `min(available cores, 4)` threads with frame and
//! slice threading. Frame threading adds `threads - 1` frames of decode
//! latency; pass `threads = 1` for the lowest latency.

mod pool;
mod sys;

use std::collections::VecDeque;
use std::ffi::{CStr, c_int, c_void};

use camtensor_core::{
    Codec, ColorMatrix, ColorRange, ColorSpec, Device, EncodedPacket, Error, Frame, FrameData,
    FrameDesc, PixelFormat, Result, StreamParams,
};

use crate::{Decoder, DecoderKind};
use pool::BufPool;
use sys::{AVCodecContext, AVFrame, AVFrameHead, AVPacket, Api};

/// Upper bound of the automatic thread count (see module docs).
const MAX_AUTO_THREADS: usize = 4;
/// Pending `emit = false` timestamps remembered at most.
const MAX_SUPPRESSED: usize = 64;
/// Buffers kept for recycling.
const POOL_SIZE: usize = 4;

/// Returns a short description of the loaded libavcodec (version and path),
/// loading it if needed.
pub fn library_info() -> Result<String> {
    let api = sys::api()?;
    Ok(format!(
        "libavcodec {} / libavutil {} ({})",
        api.avcodec_major, api.avutil_major, api.location
    ))
}

/// libavcodec software decoder.
pub struct FfmpegDecoder {
    api: &'static Api,
    ctx: *mut AVCodecContext,
    frame: *mut AVFrame,
    pkt: *mut AVPacket,
    codec: Codec,
    /// pts of packets decoded with `emit = false` whose frame has not come
    /// out yet.
    suppressed: VecDeque<i64>,
    pool: BufPool,
    last_pts: i64,
}

// SAFETY: the libavcodec context, frame and packet are owned exclusively by
// this struct and only touched through `&mut self`; libavcodec allows a
// context to be used from any thread as long as calls are not concurrent.
// `Api` only holds immutable function pointers.
unsafe impl Send for FfmpegDecoder {}

impl std::fmt::Debug for FfmpegDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfmpegDecoder")
            .field("codec", &self.codec)
            .field("avcodec_major", &self.api.avcodec_major)
            .finish_non_exhaustive()
    }
}

impl FfmpegDecoder {
    /// Opens a decoder for `params` using `threads` worker threads
    /// (0 = automatic, see the module docs).
    pub fn open(params: &StreamParams, threads: usize) -> Result<Self> {
        let api = sys::api()?;
        // Codec ids are only a fallback: they are stable for libavcodec
        // 59-62 but HEVC's id moves in 63, names never do.
        let (name, id): (&CStr, c_int) = match params.codec {
            Codec::H264 => (c"h264", 27),
            Codec::H265 => (c"hevc", 173),
            Codec::Mjpeg => (c"mjpeg", 7),
        };
        // SAFETY: NUL-terminated name; returns NULL if not found.
        let mut codec = unsafe { (api.avcodec_find_decoder_by_name)(name.as_ptr()) };
        if codec.is_null() && api.avcodec_major <= 62 {
            // SAFETY: plain lookup by id, returns NULL if not found.
            codec = unsafe { (api.avcodec_find_decoder)(id) };
        }
        if codec.is_null() {
            return Err(Error::unavailable(format!(
                "{} has no {} decoder",
                api.location,
                name.to_string_lossy()
            )));
        }
        // SAFETY: `codec` is a valid decoder returned by libavcodec.
        let ctx = unsafe { (api.avcodec_alloc_context3)(codec) };
        if ctx.is_null() {
            return Err(Error::decode("avcodec_alloc_context3 failed"));
        }
        // From here on `Drop` releases whatever was allocated.
        let mut dec = Self {
            api,
            ctx,
            frame: std::ptr::null_mut(),
            pkt: std::ptr::null_mut(),
            codec: params.codec,
            suppressed: VecDeque::new(),
            pool: BufPool::new(POOL_SIZE),
            last_pts: 0,
        };

        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get().min(MAX_AUTO_THREADS))
        } else {
            threads
        };
        dec.set_opt_int(c"threads", threads as i64)?;
        dec.set_opt_str(c"thread_type", c"frame+slice")?;

        // SAFETY: `ctx` was allocated for `codec` and is not open yet; no
        // options dictionary is passed.
        let r = unsafe { (api.avcodec_open2)(ctx, codec, std::ptr::null_mut()) };
        if r < 0 {
            return Err(Error::decode(format!(
                "avcodec_open2({}): {}",
                name.to_string_lossy(),
                api.err_str(r)
            )));
        }
        // SAFETY: allocation functions without preconditions.
        unsafe {
            dec.frame = (api.av_frame_alloc)();
            dec.pkt = (api.av_packet_alloc)();
        }
        if dec.frame.is_null() || dec.pkt.is_null() {
            return Err(Error::decode("av_frame_alloc / av_packet_alloc failed"));
        }
        tracing::debug!(codec = %params.codec, threads, lib = %api.location, "opened libavcodec decoder");
        Ok(dec)
    }

    fn set_opt_int(&mut self, name: &CStr, v: i64) -> Result<()> {
        // SAFETY: `ctx` is a valid AVCodecContext (an AVClass-enabled
        // object); the option name is NUL-terminated.
        let r = unsafe { (self.api.av_opt_set_int)(self.ctx.cast(), name.as_ptr(), v, 0) };
        if r < 0 {
            return Err(Error::decode(format!(
                "av_opt_set_int({}): {}",
                name.to_string_lossy(),
                self.api.err_str(r)
            )));
        }
        Ok(())
    }

    fn set_opt_str(&mut self, name: &CStr, v: &CStr) -> Result<()> {
        // SAFETY: as in `set_opt_int`; both strings are NUL-terminated.
        let r = unsafe { (self.api.av_opt_set)(self.ctx.cast(), name.as_ptr(), v.as_ptr(), 0) };
        if r < 0 {
            return Err(Error::decode(format!(
                "av_opt_set({}): {}",
                name.to_string_lossy(),
                self.api.err_str(r)
            )));
        }
        Ok(())
    }

    fn get_opt_int(&self, name: &CStr) -> Option<i64> {
        let mut v = 0i64;
        // SAFETY: valid context and NUL-terminated name; `v` is a valid
        // out-pointer.
        let r = unsafe {
            (self.api.av_opt_get_int)(self.ctx.cast::<c_void>(), name.as_ptr(), 0, &mut v)
        };
        (r >= 0).then_some(v)
    }

    /// Errors libavcodec reports for a single bad packet / picture. Anything
    /// else (out of memory, internal bugs, external library failures) is
    /// fatal.
    fn is_fatal(code: c_int) -> bool {
        const ENOMEM: c_int = -12;
        const BUG: c_int = sys::fferrtag(b'B', b'U', b'G', b'!');
        const BUG2: c_int = sys::fferrtag(b'B', b'U', b'G', b' ');
        const EXTERNAL: c_int = sys::fferrtag(b'E', b'X', b'T', b' ');
        matches!(code, ENOMEM | BUG | BUG2 | EXTERNAL)
    }

    /// Sends the packet currently held in `self.pkt` (or NULL to drain) and
    /// collects the frames that became available.
    fn send(&mut self, pkt: *const AVPacket, sink: &mut dyn FnMut(Frame)) -> Result<()> {
        loop {
            // SAFETY: `ctx` is open; `pkt` is either NULL (drain request)
            // or our own packet filled by `av_new_packet`.
            let r = unsafe { (self.api.avcodec_send_packet)(self.ctx, pkt) };
            match r {
                0 => break,
                sys::AVERROR_EAGAIN => {
                    // Output queue full: read frames, then retry.
                    if self.receive(sink)? == 0 {
                        return Err(Error::decode(
                            "avcodec_send_packet: EAGAIN with no frame available",
                        ));
                    }
                }
                sys::AVERROR_EOF if pkt.is_null() => break,
                e if !Self::is_fatal(e) && e != sys::AVERROR_EOF => {
                    // Typically AVERROR_INVALIDDATA (corrupt / truncated
                    // packet): drop it and keep the decoder alive.
                    let what =
                        if e == sys::AVERROR_INVALIDDATA { "corrupt" } else { "undecodable" };
                    tracing::debug!(codec = %self.codec, error = %self.api.err_str(e), "dropping {what} packet");
                    break;
                }
                e => {
                    return Err(Error::decode(format!(
                        "avcodec_send_packet: {}",
                        self.api.err_str(e)
                    )));
                }
            }
        }
        self.receive(sink).map(|_| ())
    }

    /// Receives all available frames. Returns how many results (frames or
    /// per-picture errors) were consumed.
    fn receive(&mut self, sink: &mut dyn FnMut(Frame)) -> Result<usize> {
        let mut n = 0usize;
        loop {
            // SAFETY: `ctx` is open and `frame` is a valid, unreferenced
            // AVFrame owned by us.
            let r = unsafe { (self.api.avcodec_receive_frame)(self.ctx, self.frame) };
            if r == 0 {
                n += 1;
                let res = self.handle_frame(sink);
                // SAFETY: drops the references libavcodec gave us.
                unsafe { (self.api.av_frame_unref)(self.frame) };
                res?;
            } else if r == sys::AVERROR_EAGAIN || r == sys::AVERROR_EOF {
                return Ok(n);
            } else if !Self::is_fatal(r) {
                n += 1;
                tracing::debug!(codec = %self.codec, error = %self.api.err_str(r), "picture decode error");
                if n > 4096 {
                    return Err(Error::decode("avcodec_receive_frame keeps failing"));
                }
            } else {
                return Err(Error::decode(format!(
                    "avcodec_receive_frame: {}",
                    self.api.err_str(r)
                )));
            }
        }
    }

    fn frame_pts(&self) -> i64 {
        // SAFETY: `frame` points to a live AVFrame; the prefix structs model
        // its leading fields for the loaded libavutil major (see `sys`).
        let pts = unsafe {
            if self.api.avutil_major >= 60 {
                (*self.frame.cast::<sys::AVFrameTs60>()).pts
            } else {
                (*self.frame.cast::<sys::AVFrameTs57>()).pts
            }
        };
        if pts == sys::AV_NOPTS_VALUE { self.last_pts } else { pts }
    }

    fn map_format(&self, f: c_int) -> Option<(PixelFormat, bool)> {
        let p = &self.api.pix;
        let m = if f < 0 {
            return None;
        } else if f == p.yuv420p {
            (PixelFormat::Yuv420p, false)
        } else if f == p.yuvj420p {
            (PixelFormat::Yuv420p, true)
        } else if f == p.yuv422p {
            (PixelFormat::Yuv422p, false)
        } else if f == p.yuvj422p {
            (PixelFormat::Yuv422p, true)
        } else if f == p.yuv444p {
            (PixelFormat::Yuv444p, false)
        } else if f == p.yuvj444p {
            (PixelFormat::Yuv444p, true)
        } else if f == p.nv12 {
            (PixelFormat::Nv12, false)
        } else if f == p.gray {
            (PixelFormat::Gray8, false)
        } else {
            return None;
        };
        Some(m)
    }

    /// Colour description of the current output.
    ///
    /// Per-frame `AVFrame::color_range` / `colorspace` are not read: their
    /// offsets changed between FFmpeg 5 and 8. The codec context exposes the
    /// same values (from the SPS VUI / JFIF headers) through the stable
    /// AVOption names `color_range` and `colorspace`.
    // TODO: switch to the per-frame fields once FFmpeg < 8 support is dropped.
    fn color_spec(&self, full_fmt: bool, height: u32) -> ColorSpec {
        if self.codec == Codec::Mjpeg {
            // JFIF: full-range BT.601.
            return ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt601 };
        }
        let range = if full_fmt || self.get_opt_int(c"color_range") == Some(sys::AVCOL_RANGE_JPEG) {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        let matrix = match self.get_opt_int(c"colorspace") {
            Some(sys::AVCOL_SPC_BT709) => ColorMatrix::Bt709,
            Some(sys::AVCOL_SPC_BT470BG | sys::AVCOL_SPC_SMPTE170M) => ColorMatrix::Bt601,
            _ if height >= 720 => ColorMatrix::Bt709,
            _ => ColorMatrix::Bt601,
        };
        ColorSpec { range, matrix }
    }

    fn handle_frame(&mut self, sink: &mut dyn FnMut(Frame)) -> Result<()> {
        let pts = self.frame_pts();
        if let Some(i) = self.suppressed.iter().position(|&p| p == pts) {
            self.suppressed.remove(i);
            return Ok(());
        }
        // SAFETY: `frame` holds a decoded picture; `AVFrameHead` mirrors its
        // version-stable leading fields.
        let head = unsafe { &*self.frame.cast::<AVFrameHead>() };
        let (w, h) = (head.width, head.height);
        if w <= 0 || h <= 0 {
            return Err(Error::decode(format!("invalid frame size {w}x{h}")));
        }
        let (w, h) = (w as u32, h as u32);
        let Some((format, full)) = self.map_format(head.format) else {
            return Err(Error::decode(format!(
                "unsupported pixel format {} ({w}x{h}); supported: 8-bit yuv420p, yuv422p, \
                 yuv444p, nv12, gray",
                self.api.pix_fmt_name(head.format)
            )));
        };
        let desc = FrameDesc::packed(w, h, format, self.color_spec(full, h));
        let mut buf = self.pool.take(desc.byte_len());
        copy_planes(head, &desc, &mut buf)?;
        sink(Frame { desc, data: FrameData::Host(self.pool.share(buf)), pts });
        Ok(())
    }
}

/// Copies the planes of a decoded picture into a tightly packed buffer laid
/// out as `desc`.
fn copy_planes(head: &AVFrameHead, desc: &FrameDesc, buf: &mut [u8]) -> Result<()> {
    let (w, h) = (desc.width as usize, desc.height as usize);
    let (sx, sy) = desc.format.chroma_shift();
    let (cw, ch) = ((w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy);
    for p in 0..desc.format.plane_count() {
        let (row_bytes, rows) = match (desc.format, p) {
            (_, 0) => (w, h),
            (PixelFormat::Nv12, _) => (cw * 2, ch),
            _ => (cw, ch),
        };
        let src = head.data[p];
        let ls = head.linesize[p] as isize;
        if src.is_null() || ls.unsigned_abs() < row_bytes {
            return Err(Error::decode(format!("bad plane {p} (linesize {ls}, need {row_bytes})")));
        }
        let (offset, pitch) = (desc.planes[p].offset, desc.planes[p].pitch);
        let dst = &mut buf[offset..offset + pitch * (rows - 1) + row_bytes];
        if ls == row_bytes as isize && pitch == row_bytes {
            // SAFETY: plane `p` holds `rows` rows of `linesize` bytes
            // starting at `data[p]` (libavcodec guarantee), contiguous here.
            let s = unsafe { std::slice::from_raw_parts(src, row_bytes * rows) };
            dst.copy_from_slice(s);
            continue;
        }
        for r in 0..rows {
            // SAFETY: row `r` of plane `p` starts at `data[p] + r * linesize`
            // and has at least `row_bytes` (<= |linesize|) valid bytes.
            let s = unsafe { std::slice::from_raw_parts(src.offset(r as isize * ls), row_bytes) };
            dst[r * pitch..r * pitch + row_bytes].copy_from_slice(s);
        }
    }
    Ok(())
}

impl Decoder for FfmpegDecoder {
    fn decode(
        &mut self,
        pkt: &EncodedPacket,
        emit: bool,
        sink: &mut dyn FnMut(Frame),
    ) -> Result<()> {
        if pkt.data.is_empty() {
            // An empty packet would be interpreted as a drain request.
            return Ok(());
        }
        if !emit && self.codec == Codec::Mjpeg {
            // Intra-only: nothing depends on this picture.
            return Ok(());
        }
        let len = c_int::try_from(pkt.data.len())
            .map_err(|_| Error::decode("packet larger than 2 GiB"))?;
        // SAFETY: `self.pkt` is our unreferenced packet; av_new_packet
        // allocates `len` bytes plus zeroed input padding.
        let r = unsafe { (self.api.av_new_packet)(self.pkt, len) };
        if r < 0 {
            return Err(Error::decode(format!("av_new_packet: {}", self.api.err_str(r))));
        }
        // SAFETY: `data` points to at least `len` writable bytes (just
        // allocated), the source slice has exactly `len` bytes; the leading
        // AVPacket fields are version-stable (see `sys`).
        unsafe {
            let p = &mut *self.pkt;
            std::ptr::copy_nonoverlapping(pkt.data.as_ptr(), p.data, pkt.data.len());
            p.pts = pkt.pts;
            p.dts = sys::AV_NOPTS_VALUE;
            if pkt.keyframe {
                p.flags |= sys::AV_PKT_FLAG_KEY;
            }
        }
        self.last_pts = pkt.pts;
        if !emit {
            if self.suppressed.len() >= MAX_SUPPRESSED {
                self.suppressed.pop_front();
            }
            self.suppressed.push_back(pkt.pts);
        }
        let res = self.send(self.pkt, sink);
        // SAFETY: releases our reference to the packet buffer (libavcodec
        // holds its own if it kept the data).
        unsafe { (self.api.av_packet_unref)(self.pkt) };
        res
    }

    fn flush(&mut self, sink: &mut dyn FnMut(Frame)) -> Result<()> {
        let res = self.send(std::ptr::null(), sink);
        // SAFETY: `ctx` is open; resets the decoder (also leaves draining
        // mode) so it can be fed again.
        unsafe { (self.api.avcodec_flush_buffers)(self.ctx) };
        self.suppressed.clear();
        res
    }

    fn kind(&self) -> DecoderKind {
        DecoderKind::Ffmpeg
    }

    fn device(&self) -> Device {
        Device::Cpu
    }
}

impl Drop for FfmpegDecoder {
    fn drop(&mut self) {
        // SAFETY: each pointer is either NULL (the free functions accept
        // that) or was allocated by the matching libav* allocator and is
        // owned by us; the free functions reset them to NULL.
        unsafe {
            (self.api.av_frame_free)(&mut self.frame);
            (self.api.av_packet_free)(&mut self.pkt);
            (self.api.avcodec_free_context)(&mut self.ctx);
        }
    }
}
