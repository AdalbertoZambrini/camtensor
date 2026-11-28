//! Shared helpers for the CPU decoder integration tests: locating the
//! FFmpeg CLI and shared libraries, generating test media, and feeding
//! decoders.

#![allow(dead_code)]

pub mod annexb;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, Once, OnceLock};

use bytes::Bytes;
use camtensor_core::{Codec, EncodedPacket, Frame, Result};
use camtensor_decode::Decoder;

/// FFmpeg shared libraries on the development box (used when
/// `CAMTENSOR_FFMPEG_DIR` is unset).
const DEFAULT_FFMPEG_DIR: &str =
    r"D:\github\portfolio\.devdeps\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1\bin";
/// Static FFmpeg CLI with libx264/libx265 on the development box.
const DEFAULT_FFMPEG_CLI: &str = r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe";

static ENV_INIT: Once = Once::new();

/// Points the decoder at the FFmpeg libraries; returns `false` (after
/// printing "skipped") when they cannot be loaded.
#[cfg(feature = "ffmpeg")]
pub fn ffmpeg_libs(test: &str) -> bool {
    ENV_INIT.call_once(|| {
        if std::env::var_os("CAMTENSOR_FFMPEG_DIR").is_none()
            && Path::new(DEFAULT_FFMPEG_DIR).is_dir()
        {
            // SAFETY: executed once, before any test in this binary loads
            // libavcodec (every FFmpeg test passes through this `Once`), so
            // no foreign code reads the environment concurrently.
            unsafe { std::env::set_var("CAMTENSOR_FFMPEG_DIR", DEFAULT_FFMPEG_DIR) };
        }
    });
    match camtensor_decode::ffmpeg::library_info() {
        Ok(info) => {
            eprintln!("{test}: using {info}");
            true
        }
        Err(e) => {
            eprintln!("{test}: skipped (libavcodec unavailable: {e})");
            false
        }
    }
}

/// Path of the FFmpeg command-line tool, if any.
pub fn ffmpeg_cli() -> Option<&'static Path> {
    static CLI: OnceLock<Option<PathBuf>> = OnceLock::new();
    CLI.get_or_init(|| {
        let mut cands = Vec::new();
        if let Some(p) = std::env::var_os("CAMTENSOR_FFMPEG") {
            cands.push(PathBuf::from(p));
        }
        cands.push(PathBuf::from(DEFAULT_FFMPEG_CLI));
        cands.push(PathBuf::from("ffmpeg"));
        cands.into_iter().find(|p| {
            Command::new(p).arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
        })
    })
    .as_deref()
}

/// Encoded test clip, split into packets with presentation timestamps.
#[derive(Debug)]
pub struct Media {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub packets: Vec<EncodedPacket>,
}

impl Media {
    /// Input pts in presentation order.
    pub fn sorted_pts(&self) -> Vec<i64> {
        let mut v: Vec<i64> = self.packets.iter().map(|p| p.pts).collect();
        v.sort_unstable();
        v
    }

    /// `true` if packets are not in presentation order (B-frames).
    pub fn has_reordering(&self) -> bool {
        self.packets.windows(2).any(|w| w[1].pts < w[0].pts)
    }
}

/// Description of a clip to generate.
#[derive(Debug, Clone)]
pub struct Spec {
    /// Unique name (cache key and file stem).
    pub name: &'static str,
    pub codec: Codec,
    /// lavfi source graph, e.g. `testsrc2=size=1280x720:rate=25`.
    pub source: String,
    pub frames: u32,
    /// Encoder arguments (`-c:v ...`).
    pub encoder: Vec<String>,
}

impl Spec {
    pub fn new(name: &'static str, codec: Codec, source: &str, frames: u32, enc: &str) -> Self {
        Self {
            name,
            codec,
            source: source.to_string(),
            frames,
            encoder: enc.split_whitespace().map(str::to_string).collect(),
        }
    }
}

fn work_dir() -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("camtensor-decode")
        .join(env!("CARGO_CRATE_NAME"));
    std::fs::create_dir_all(&d).expect("create test media dir");
    d
}

fn run(cmd: &mut Command) -> std::result::Result<Vec<u8>, String> {
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(out.stdout)
}

/// Generates (once per test binary) and returns the clip, or `None` with a
/// "skipped" message when the FFmpeg CLI is missing or cannot produce it.
pub fn media(test: &str, spec: &Spec) -> Option<Arc<Media>> {
    type Cache = Mutex<HashMap<&'static str, Option<Arc<Media>>>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let Some(cli) = ffmpeg_cli() else {
        eprintln!("{test}: skipped (ffmpeg CLI not found; set CAMTENSOR_FFMPEG)");
        return None;
    };
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    let m = cache
        .entry(spec.name)
        .or_insert_with(|| match generate(cli, spec) {
            Ok(m) => Some(Arc::new(m)),
            Err(e) => {
                eprintln!("could not generate {}: {e}", spec.name);
                None
            }
        })
        .clone();
    if m.is_none() {
        eprintln!("{test}: skipped (media generation failed)");
    }
    m
}

fn generate(cli: &Path, spec: &Spec) -> std::result::Result<Media, String> {
    let dir = work_dir();
    let base = |ext: &str| dir.join(format!("{}.{ext}", spec.name));
    let mut enc = Command::new(cli);
    enc.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
        .arg(&spec.source)
        .args(["-frames:v", &spec.frames.to_string()])
        .args(&spec.encoder);

    let (packets, width, height) = match spec.codec {
        Codec::RawVideo => return Err("raw video is not encoded".into()),
        Codec::Mjpeg => {
            let out = base("mjpeg");
            run(enc.args(["-f", "mjpeg"]).arg(&out))?;
            let data = std::fs::read(&out).map_err(|e| e.to_string())?;
            let jpegs = annexb::split_jpegs(&data);
            assert_eq!(jpegs.len(), spec.frames as usize, "{}: JPEGs found", spec.name);
            let (w, h) = jpeg_size(&jpegs[0]).ok_or("no SOF")?;
            let pkts = jpegs
                .into_iter()
                .enumerate()
                .map(|(i, j)| packet(spec.codec, j, i as i64 * 3600, true, i as u64))
                .collect();
            (pkts, w, h)
        }
        Codec::H264 | Codec::H265 => {
            // Encode into Matroska to obtain per-packet pts, then remux to
            // an Annex B elementary stream (parameter sets are re-inserted
            // before every keyframe by the *_mp4toannexb filter).
            let mkv = base("mkv");
            run(enc.arg(&mkv))?;
            let (std, fmt, bsf) = match spec.codec {
                Codec::H264 => (annexb::Standard::H264, "h264", "h264_mp4toannexb"),
                _ => (annexb::Standard::H265, "hevc", "hevc_mp4toannexb"),
            };
            let es = base(fmt);
            // The bitstream filter must be explicit: the raw muxers'
            // auto-detection mistakes a 256..511-byte NAL length prefix
            // (`00 00 01 xx`) for a start code.
            run(Command::new(cli)
                .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
                .arg(&mkv)
                .args(["-c", "copy", "-bsf:v", bsf, "-f", fmt])
                .arg(&es))?;
            let crc = run(Command::new(cli)
                .args(["-hide_banner", "-loglevel", "error", "-i"])
                .arg(&mkv)
                .args(["-c", "copy", "-f", "framecrc", "-"]))?;
            let pts = parse_framecrc_pts(&String::from_utf8_lossy(&crc))?;
            let data = std::fs::read(&es).map_err(|e| e.to_string())?;
            let aus = annexb::split_access_units(&data, std);
            // A mismatch is a bug in the test splitter, not a missing tool.
            assert_eq!(aus.len(), pts.len(), "{}: AUs found vs container packets", spec.name);
            assert_eq!(pts.len(), spec.frames as usize, "{}: encoded packets", spec.name);
            let pkts = aus
                .into_iter()
                .zip(pts)
                .enumerate()
                .map(|(i, (au, pts))| packet(spec.codec, au.data, pts, au.keyframe, i as u64))
                .collect();
            let (w, h) = parse_size(&spec.source).ok_or("no size in source")?;
            (pkts, w, h)
        }
    };
    Ok(Media { codec: spec.codec, width, height, packets })
}

/// Builds an [`EncodedPacket`].
pub fn packet(codec: Codec, data: Vec<u8>, pts: i64, keyframe: bool, seq: u64) -> EncodedPacket {
    EncodedPacket {
        stream_id: Arc::from("test"),
        codec,
        data: Bytes::from(data),
        pts,
        keyframe,
        seq,
        recv_time_ns: 0,
        capture_time_ns: None,
        loss: false,
    }
}

/// Packet pts (converted to 90 kHz) from `-f framecrc` output, in decode
/// order.
fn parse_framecrc_pts(s: &str) -> std::result::Result<Vec<i64>, String> {
    let mut tb = (1i64, 1000i64);
    let mut out = Vec::new();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("#tb 0:") {
            let (n, d) = rest.trim().split_once('/').ok_or("bad #tb")?;
            tb = (n.parse().map_err(|_| "bad tb")?, d.parse().map_err(|_| "bad tb")?);
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            let pts: i64 = f.get(2).and_then(|v| v.parse().ok()).ok_or("bad framecrc line")?;
            out.push(pts * 90_000 * tb.0 / tb.1);
        }
    }
    Ok(out)
}

fn parse_size(source: &str) -> Option<(u32, u32)> {
    let s = source.split("size=").nth(1)?;
    let s: String = s.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
    let (w, h) = s.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// Width/height from the SOF marker of a JPEG.
fn jpeg_size(j: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2;
    while i + 9 < j.len() {
        if j[i] != 0xff {
            return None;
        }
        let m = j[i + 1];
        let len = u16::from_be_bytes([j[i + 2], j[i + 3]]) as usize;
        if matches!(m, 0xc0..=0xc3) {
            let h = u16::from_be_bytes([j[i + 5], j[i + 6]]) as u32;
            let w = u16::from_be_bytes([j[i + 7], j[i + 8]]) as u32;
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}

/// Feeds all packets (`emit(i)` decides per packet), then flushes.
pub fn decode_all(
    dec: &mut dyn Decoder,
    packets: &[EncodedPacket],
    emit: impl Fn(usize) -> bool,
) -> Result<Vec<Frame>> {
    let mut frames = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        dec.decode(p, emit(i), &mut |f| frames.push(f))?;
    }
    dec.flush(&mut |f| frames.push(f))?;
    Ok(frames)
}

/// Mean value of each plane of a host frame.
pub fn plane_means(f: &Frame) -> Vec<f64> {
    let data = f.host_bytes().expect("host frame");
    let d = &f.desc;
    let (sx, sy) = d.format.chroma_shift();
    (0..d.format.plane_count())
        .map(|p| {
            let (w, h) = if p == 0 {
                let bpp =
                    if matches!(d.format, camtensor_core::PixelFormat::Rgb24) { 3 } else { 1 };
                (d.width as usize * bpp, d.height as usize)
            } else {
                let cw = (d.width as usize).div_ceil(1 << sx);
                let ch = (d.height as usize).div_ceil(1 << sy);
                let bpp = if d.format == camtensor_core::PixelFormat::Nv12 { 2 } else { 1 };
                (cw * bpp, ch)
            };
            let pl = d.planes[p];
            let mut sum = 0u64;
            for r in 0..h {
                let row = &data[pl.offset + r * pl.pitch..][..w];
                sum += row.iter().map(|&v| v as u64).sum::<u64>();
            }
            sum as f64 / (w * h) as f64
        })
        .collect()
}

/// Mean R, G, B of a packed RGB24 frame.
pub fn rgb_means(f: &Frame) -> [f64; 3] {
    let data = f.host_bytes().expect("host frame");
    let mut s = [0u64; 3];
    for px in data[..f.desc.byte_len()].as_chunks::<3>().0 {
        for (acc, &v) in s.iter_mut().zip(px) {
            *acc += v as u64;
        }
    }
    let n = (f.desc.width * f.desc.height) as f64;
    [s[0] as f64 / n, s[1] as f64 / n, s[2] as f64 / n]
}
