//! Capture-device source (webcams, USB/HDMI capture cards) through
//! libavdevice: `dshow` on Windows, `v4l2` on Linux, `avfoundation` on macOS.
//!
//! MJPEG devices produce packets that go through the normal decode path
//! (NVDEC or CPU). Uncompressed devices (YUYV, UYVY, NV12, I420, grey) are
//! repacked to a planar layout here and delivered as frames, without a
//! decoder.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use bytes::Bytes;

use camtensor_core::time::{PTS_CLOCK_HZ, now_ns};
use camtensor_core::{
    CaptureConfig, Codec, ColorMatrix, ColorRange, ColorSpec, EncodedPacket, Error, Frame,
    FrameData, FrameDesc, PixelFormat, Result, StreamParams,
};
use camtensor_decode::ffmpeg::demux::{self, Demuxer, RawVideoInfo};
use camtensor_ingest::{RawFrame, SourceEvent};

/// `AVPixelFormat` ids (stable across FFmpeg 5–8).
const AV_PIX_FMT_YUV420P: i32 = 0;
const AV_PIX_FMT_YUYV422: i32 = 1;
const AV_PIX_FMT_GRAY8: i32 = 8;
const AV_PIX_FMT_YUVJ420P: i32 = 12;
const AV_PIX_FMT_UYVY422: i32 = 15;
const AV_PIX_FMT_NV12: i32 = 23;

/// Handle of a running capture source.
pub(crate) struct CaptureSource {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CaptureSource {
    pub(crate) fn spawn(
        stream_id: Arc<str>,
        device: String,
        cfg: CaptureConfig,
        tx: tokio::sync::mpsc::Sender<SourceEvent>,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let thread = std::thread::Builder::new()
            .name(format!("ct-capture-{stream_id}"))
            .spawn(move || {
                if let Err(e) = run(&stream_id, &device, &cfg, &tx, &s) {
                    tracing::warn!(stream = %stream_id, "capture: {e}");
                    let _ = tx.blocking_send(SourceEvent::Disconnected { reason: e.to_string() });
                }
            })
            .map_err(Error::Io)?;
        Ok(Self { stop, thread: Some(thread) })
    }

    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for CaptureSource {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Backend-specific device URL and demuxer options.
fn open_args(device: &str, cfg: &CaptureConfig) -> (String, String, Vec<(String, String)>) {
    let backend = cfg.backend.clone().unwrap_or_else(|| demux::default_capture_backend().into());
    let url = if backend == "dshow" && !device.contains('=') {
        format!("video={device}")
    } else {
        device.to_string()
    };
    let mut opts = Vec::new();
    if let (Some(w), Some(h)) = (cfg.width, cfg.height) {
        opts.push(("video_size".into(), format!("{w}x{h}")));
    }
    if let Some(f) = cfg.fps {
        opts.push(("framerate".into(), format!("{f}")));
    }
    if let Some(fmt) = &cfg.format {
        let key = match (backend.as_str(), fmt.as_str()) {
            ("dshow", "mjpeg" | "h264") => "vcodec",
            ("dshow", _) => "pixel_format",
            ("v4l2", _) => "input_format",
            _ => "pixel_format",
        };
        opts.push((key.into(), fmt.clone()));
    }
    if backend == "dshow" {
        // Avoid "real-time buffer full" frame drops at high resolutions.
        opts.push(("rtbufsize".into(), "64M".into()));
    }
    (backend, url, opts)
}

/// Lists capture devices of the platform's backend (or `backend`).
pub fn list_devices(backend: Option<&str>) -> Result<Vec<(String, String)>> {
    demux::list_devices(backend.unwrap_or(demux::default_capture_backend()))
}

fn run(
    stream_id: &Arc<str>,
    device: &str,
    cfg: &CaptureConfig,
    tx: &tokio::sync::mpsc::Sender<SourceEvent>,
    stop: &AtomicBool,
) -> Result<()> {
    let (backend, url, opts) = open_args(device, cfg);
    let mut dmx = Demuxer::open_device(&backend, &url, &opts)?;
    let info = dmx.info().clone();
    let send = |ev: SourceEvent| tx.blocking_send(ev).map_err(|_| Error::Closed);
    let (width, height) = info.raw.map(|r| (r.width, r.height)).unwrap_or((0, 0));
    if !matches!(info.codec, Codec::Mjpeg | Codec::RawVideo) {
        return Err(Error::Unavailable(format!(
            "{device}: compressed capture format {} is not supported (use mjpeg or a raw format)",
            info.codec
        )));
    }
    if let Some(r) = info.raw
        && info.codec == Codec::RawVideo
    {
        raw_format(r.pix_fmt).ok_or_else(|| {
            Error::Unavailable(format!(
                "{device}: unsupported raw pixel format id {} (supported: yuyv422, uyvy422, nv12, yuv420p, gray)",
                r.pix_fmt
            ))
        })?;
    }

    send(SourceEvent::Connected(StreamParams {
        codec: info.codec,
        width,
        height,
        extradata: Bytes::new(),
        fps: info.fps,
    }))?;
    tracing::info!(stream = %stream_id, %backend, device = %url, codec = %info.codec, "{width}x{height} capture started");

    let t0 = Instant::now();
    let mut seq = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let Some(p) = dmx.read()? else {
            send(SourceEvent::Disconnected { reason: "capture ended".into() })?;
            return Ok(());
        };
        let recv = now_ns();
        let pts = (t0.elapsed().as_secs_f64() * PTS_CLOCK_HZ as f64) as i64;
        let ev = match (info.codec, info.raw) {
            (Codec::RawVideo, Some(r)) => match to_frame(&p.data, r, pts) {
                Some(frame) => SourceEvent::Frame(RawFrame { frame, seq, recv_time_ns: recv }),
                None => {
                    tracing::debug!(stream = %stream_id, "short raw frame ({} bytes)", p.data.len());
                    continue;
                }
            },
            _ => SourceEvent::Packet(EncodedPacket {
                stream_id: stream_id.clone(),
                codec: info.codec,
                data: Bytes::from(p.data),
                pts,
                keyframe: true,
                seq,
                recv_time_ns: recv,
                capture_time_ns: None,
                loss: false,
            }),
        };
        seq += 1;
        // Blocking send: the device buffers while the pipeline catches up;
        // dshow/v4l2 drop at the source if it falls too far behind.
        send(ev)?;
    }
    Ok(())
}

fn raw_format(pix_fmt: i32) -> Option<(PixelFormat, ColorRange)> {
    Some(match pix_fmt {
        AV_PIX_FMT_YUYV422 | AV_PIX_FMT_UYVY422 => (PixelFormat::Yuv422p, ColorRange::Limited),
        AV_PIX_FMT_NV12 => (PixelFormat::Nv12, ColorRange::Limited),
        AV_PIX_FMT_YUV420P => (PixelFormat::Yuv420p, ColorRange::Limited),
        AV_PIX_FMT_YUVJ420P => (PixelFormat::Yuv420p, ColorRange::Full),
        AV_PIX_FMT_GRAY8 => (PixelFormat::Gray8, ColorRange::Full),
        _ => return None,
    })
}

/// Converts one raw device frame to a planar host frame.
fn to_frame(data: &[u8], r: RawVideoInfo, pts: i64) -> Option<Frame> {
    let (format, range) = raw_format(r.pix_fmt)?;
    let (w, h) = (r.width as usize, r.height as usize);
    // Webcams are SD/HD sources; BT.601 is the common default for YUY2/NV12.
    let color =
        ColorSpec { range, matrix: if h >= 720 { ColorMatrix::Bt709 } else { ColorMatrix::Bt601 } };
    let desc = FrameDesc::packed(r.width, r.height, format, color);
    let mut out = vec![0u8; desc.byte_len()];
    match r.pix_fmt {
        AV_PIX_FMT_YUYV422 | AV_PIX_FMT_UYVY422 => {
            let row = w * 2;
            if data.len() < row * h || w % 2 != 0 {
                return None;
            }
            // YUYV: [Y0 U Y1 V]; UYVY: [U Y0 V Y1].
            let y_off = if r.pix_fmt == AV_PIX_FMT_YUYV422 { 0 } else { 1 };
            let cw = w / 2;
            let (yp, rest) = out.split_at_mut(w * h);
            let (up, vp) = rest.split_at_mut(cw * h);
            for (y, src) in data.chunks_exact(row).take(h).enumerate() {
                let yrow = &mut yp[y * w..(y + 1) * w];
                let urow = &mut up[y * cw..(y + 1) * cw];
                let vrow = &mut vp[y * cw..(y + 1) * cw];
                for (x, px) in src.as_chunks::<4>().0.iter().enumerate() {
                    yrow[2 * x] = px[y_off];
                    yrow[2 * x + 1] = px[y_off + 2];
                    urow[x] = px[1 - y_off];
                    vrow[x] = px[3 - y_off];
                }
            }
        }
        _ => {
            if data.len() < out.len() {
                return None;
            }
            let n = out.len();
            out.copy_from_slice(&data[..n]);
        }
    }
    Some(Frame { desc, data: FrameData::Host(Arc::new(out)), pts })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_and_uyvy_repack() {
        // 2x1 image: YUYV = Y0=10 U=20 Y1=30 V=40.
        let r = RawVideoInfo { width: 2, height: 1, pix_fmt: AV_PIX_FMT_YUYV422 };
        let f = to_frame(&[10, 20, 30, 40], r, 0).unwrap();
        assert_eq!(f.desc.format, PixelFormat::Yuv422p);
        assert_eq!(f.host_bytes().unwrap(), &[10, 30, 20, 40]);
        let r = RawVideoInfo { width: 2, height: 1, pix_fmt: AV_PIX_FMT_UYVY422 };
        let f = to_frame(&[20, 10, 40, 30], r, 0).unwrap();
        assert_eq!(f.host_bytes().unwrap(), &[10, 30, 20, 40]);
        // Short buffers are rejected.
        assert!(to_frame(&[1, 2, 3], r, 0).is_none());
        // Unsupported formats.
        assert!(to_frame(&[0; 16], RawVideoInfo { width: 2, height: 2, pix_fmt: 99 }, 0).is_none());
    }

    #[test]
    fn dshow_and_v4l2_arguments() {
        let cfg = CaptureConfig {
            backend: Some("dshow".into()),
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            format: Some("mjpeg".into()),
        };
        let (b, url, opts) = open_args("USB Camera", &cfg);
        assert_eq!((b.as_str(), url.as_str()), ("dshow", "video=USB Camera"));
        assert!(opts.contains(&("video_size".into(), "1280x720".into())));
        assert!(opts.contains(&("vcodec".into(), "mjpeg".into())));
        let cfg = CaptureConfig {
            backend: Some("v4l2".into()),
            format: Some("yuyv422".into()),
            ..Default::default()
        };
        let (_, url, opts) = open_args("/dev/video0", &cfg);
        assert_eq!(url, "/dev/video0");
        assert!(opts.contains(&("input_format".into(), "yuyv422".into())));
    }
}
