//! File sources, regions of interest and JPEG snapshots.
//!
//! Clips are generated with FFmpeg into a temp dir; tests skip when FFmpeg
//! (CLI or shared libraries) is not available.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use camtensor::{
    DType, Device, Engine, EngineConfig, Layout, OutputConfig, PlaybackConfig, Region, ResizeMode,
    SnapshotOptions, StreamConfig, TensorSpec,
};

static SERIAL: Mutex<()> = Mutex::new(());

fn ffmpeg() -> Option<PathBuf> {
    ffmpeg_libs();
    let cands = [
        std::env::var("CAMTENSOR_FFMPEG").ok().map(PathBuf::from),
        Some(PathBuf::from(r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe")),
        Some(PathBuf::from("/usr/bin/ffmpeg")),
    ];
    cands.into_iter().flatten().find(|p| p.exists())
}

fn ffmpeg_libs() {
    if std::env::var_os("CAMTENSOR_FFMPEG_DIR").is_none() {
        let dev = r"D:\github\portfolio\.devdeps\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1\bin";
        if Path::new(dev).is_dir() {
            // SAFETY: set before any engine thread starts.
            unsafe { std::env::set_var("CAMTENSOR_FFMPEG_DIR", dev) };
        }
    }
}

/// Generates `frames` frames of testsrc2 at 25 fps with the given encoder
/// args; `None` when this FFmpeg build lacks the encoder (e.g. LGPL builds
/// have no libx264/libx265).
fn clip(ff: &Path, name: &str, size: &str, frames: u32, args: &[&str]) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!("camtensor-features-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join(name);
    if !out.exists() {
        let st = Command::new(ff)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("testsrc2=size={size}:rate=25"))
            .args(["-frames:v", &frames.to_string()])
            .args(args)
            .arg(&out)
            .status()
            .unwrap();
        if !st.success() {
            eprintln!("skipped {name}: this ffmpeg cannot encode it");
            let _ = std::fs::remove_file(&out);
            return None;
        }
    }
    Some(out)
}

fn engine(
    url: String,
    playback: PlaybackConfig,
    region: Option<Region>,
    out: OutputConfig,
) -> Engine {
    let mut cfg = EngineConfig { device: Device::Auto, ..Default::default() };
    cfg.outputs.insert("o".into(), out);
    let mut s = StreamConfig::new("f", url, &["o"]);
    s.playback = playback;
    s.region = region;
    cfg.streams.push(s);
    Engine::new(cfg).unwrap()
}

fn out_cfg(w: u32, h: u32) -> OutputConfig {
    OutputConfig {
        tensor: TensorSpec {
            width: w,
            height: h,
            dtype: DType::U8,
            layout: Layout::Nhwc,
            ..Default::default()
        },
        keep_frame: true,
        backpressure: camtensor::Backpressure::Block,
        ..Default::default()
    }
}

fn file_url(p: &Path) -> String {
    format!("file:{}", p.display())
}

fn drain(e: &Engine, until_idle: Duration) -> Vec<camtensor::Batch> {
    // Keep only a couple of batches so the output pool is never exhausted.
    let mut v = Vec::new();
    let mut last = Instant::now();
    while last.elapsed() < until_idle {
        if let Ok(b) = e.recv("o", Some(Duration::from_millis(100))) {
            last = Instant::now();
            if v.len() < 2 {
                v.push(b);
            }
        }
    }
    v
}

#[test]
fn file_sources_deliver_every_frame() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let Some(ff) = ffmpeg() else {
        eprintln!("skipped: ffmpeg not found");
        return;
    };
    let h264_mp4 = clip(
        &ff,
        "h264.mp4",
        "640x360",
        50,
        &["-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p"],
    );
    let h265_mkv = clip(
        &ff,
        "h265.mkv",
        "640x360",
        50,
        &["-c:v", "libx265", "-g", "10", "-x265-params", "log-level=error", "-pix_fmt", "yuv420p"],
    );
    let mjpeg_mkv = clip(
        &ff,
        "mjpeg.mkv",
        "320x240",
        50,
        &["-c:v", "mjpeg", "-q:v", "5", "-pix_fmt", "yuvj420p"],
    );
    let raw_h264 = clip(
        &ff,
        "raw.h264",
        "640x360",
        50,
        &["-c:v", "libx264", "-g", "10", "-bsf:v", "h264_mp4toannexb", "-f", "h264"],
    );

    // MJPEG is always encodable (native FFmpeg encoder).
    let mjpeg_mkv = mjpeg_mkv.expect("mjpeg encoder");
    let cases = [
        (h264_mp4.clone(), 640, 360),
        (h265_mkv, 640, 360),
        (Some(mjpeg_mkv.clone()), 320, 240),
        (raw_h264, 640, 360),
    ];
    for (path, w, h) in cases.iter().filter_map(|(p, w, h)| p.as_ref().map(|p| (p, *w, *h))) {
        let e = engine(
            file_url(path),
            PlaybackConfig { realtime: false, looped: false },
            None,
            out_cfg(64, 64),
        );
        let mut n = 0;
        let mut last = Instant::now();
        let mut src = (0, 0);
        while last.elapsed() < Duration::from_secs(3) {
            if let Ok(b) = e.recv("o", Some(Duration::from_millis(100))) {
                last = Instant::now();
                n += b.valid_count();
                src = (b.items()[0].meta.src_width, b.items()[0].meta.src_height);
            }
        }
        let st = e.stats().streams[0].clone();
        assert_eq!(n, 50, "{}: every frame delivered once ({st:?})", path.display());
        assert_eq!(src, (w, h), "{}", path.display());
        assert_eq!(st.last_error.as_deref(), Some("end of file"));
        e.shutdown();
    }

    // Real-time playback paces at the file rate (25 fps): 50 frames ≈ 2 s.
    let t0 = Instant::now();
    let paced = h264_mp4.unwrap_or_else(|| mjpeg_mkv.clone());
    let e = engine(
        file_url(&paced),
        PlaybackConfig { realtime: true, looped: false },
        None,
        out_cfg(64, 64),
    );
    let mut n = 0;
    while n < 50 && t0.elapsed() < Duration::from_secs(10) {
        if let Ok(b) = e.recv("o", Some(Duration::from_millis(100))) {
            n += b.valid_count();
        }
    }
    let el = t0.elapsed();
    assert_eq!(n, 50);
    assert!(el >= Duration::from_millis(1800), "paced, took {el:?}");
    e.shutdown();

    // Looping keeps going past the end with increasing timestamps.
    let e = engine(
        file_url(&mjpeg_mkv),
        PlaybackConfig { realtime: false, looped: true },
        None,
        out_cfg(32, 32),
    );
    let (mut n, mut last_pts) = (0, i64::MIN);
    let t0 = Instant::now();
    while n < 120 && t0.elapsed() < Duration::from_secs(15) {
        if let Ok(b) = e.recv("o", Some(Duration::from_millis(100))) {
            for it in b.items().iter().filter(|i| i.valid) {
                assert!(it.meta.pts > last_pts, "pts increase across loops");
                last_pts = it.meta.pts;
                n += 1;
            }
        }
    }
    assert!(n >= 120, "looped beyond 50 frames: {n}");
    e.shutdown();

    // Missing file: clear error, no panic.
    let e =
        engine("file:does/not/exist.mp4".into(), PlaybackConfig::default(), None, out_cfg(32, 32));
    std::thread::sleep(Duration::from_millis(500));
    let err = e.stats().streams[0].last_error.clone().unwrap_or_default();
    assert!(err.contains("cannot open"), "{err}");
    e.shutdown();
}

#[test]
fn region_of_interest_and_snapshots() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let Some(ff) = ffmpeg() else {
        eprintln!("skipped: ffmpeg not found");
        return;
    };
    // H.264 when the encoder exists, MJPEG otherwise (always available).
    let path = clip(
        &ff,
        "roi.mp4",
        "1280x720",
        20,
        &["-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p"],
    )
    .or_else(|| {
        clip(
            &ff,
            "roi.mkv",
            "1280x720",
            20,
            &["-c:v", "mjpeg", "-q:v", "3", "-pix_fmt", "yuvj420p"],
        )
    })
    .expect("mjpeg encoder");
    let region = Region { x: 0.5, y: 0.0, w: 0.5, h: 1.0 };
    let mut out = out_cfg(320, 320);
    out.tensor.resize = ResizeMode::Letterbox;
    let e = engine(
        file_url(&path),
        PlaybackConfig { realtime: false, looped: false },
        Some(region),
        out,
    );
    let batches = drain(&e, Duration::from_secs(2));
    let item = batches[0].items()[0].clone();
    // The right half (640x720) letterboxed into 320x320: scale 320/720,
    // centred horizontally; tensor (0, 0) maps back into the region.
    let s = 320.0 / 720.0;
    let m = &item.meta;
    assert!((m.transform.0[0] - s).abs() < 1e-4, "{:?}", m.transform);
    let (x, y) = m.to_source(160.0, 160.0);
    assert!(
        (x - 960.0).abs() < 1.0 && (y - 360.0).abs() < 1.0,
        "centre maps to the region centre: {x},{y}"
    );
    let (x0, _) = m.to_source((320.0 - 640.0 * s) / 2.0, 0.0);
    assert!((x0 - 640.0).abs() < 1.0, "left edge of the image is the region's left edge: {x0}");

    let frame = item.frame.expect("kept frame");
    let full = e.snapshot(&frame, &SnapshotOptions::default()).unwrap();
    assert_eq!(&full[..2], &[0xFF, 0xD8], "JPEG SOI");
    assert_eq!(jpeg_size(&full), Some((1280, 720)));
    let part = e
        .snapshot(
            &frame,
            &SnapshotOptions {
                region: Some([100.0, 50.0, 400.0, 300.0]),
                width: Some(200),
                quality: 80,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(jpeg_size(&part), Some((200, 150)), "width given, height keeps the aspect");
    let clamped = e
        .snapshot(
            &frame,
            &SnapshotOptions { region: Some([1200.0, 700.0, 500.0, 500.0]), ..Default::default() },
        )
        .unwrap();
    assert_eq!(jpeg_size(&clamped), Some((80, 20)), "region clamped to the frame");
    assert!(
        e.snapshot(
            &frame,
            &SnapshotOptions { region: Some([2000.0, 0.0, 10.0, 10.0]), ..Default::default() }
        )
        .is_err()
    );
    e.shutdown();
}

/// Reads the size from a baseline/progressive JPEG SOF marker.
fn jpeg_size(j: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2;
    while i + 9 < j.len() {
        if j[i] != 0xFF {
            return None;
        }
        let marker = j[i + 1];
        let len = u16::from_be_bytes([j[i + 2], j[i + 3]]) as usize;
        if matches!(marker, 0xC0..=0xC2) {
            let h = u16::from_be_bytes([j[i + 5], j[i + 6]]) as u32;
            let w = u16::from_be_bytes([j[i + 7], j[i + 8]]) as u32;
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}

#[test]
fn webcam_if_present() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    ffmpeg_libs();
    let devices = match camtensor::list_capture_devices(None) {
        Ok(d) if !d.is_empty() => d,
        Ok(_) => {
            eprintln!("skipped: no capture device");
            return;
        }
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    let (name, desc) = &devices[0];
    let url = if cfg!(windows) && !desc.is_empty() {
        format!("device:{desc}")
    } else {
        format!("device:{name}")
    };
    let mut out = out_cfg(160, 120);
    out.backpressure = camtensor::Backpressure::DropOldest;
    let e = engine(url, PlaybackConfig::default(), None, out);
    let b = e.recv("o", Some(Duration::from_secs(20))).expect("a webcam frame");
    let item = &b.items()[0];
    assert!(item.meta.src_width > 0 && item.meta.src_height > 0);
    let jpeg = e.snapshot(item.frame.as_ref().unwrap(), &SnapshotOptions::default()).unwrap();
    assert_eq!(jpeg_size(&jpeg), Some((item.meta.src_width, item.meta.src_height)));
    let st = e.stats().streams[0].clone();
    assert!(matches!(st.codec.as_deref(), Some("rawvideo") | Some("mjpeg")), "{st:?}");
    e.shutdown();
}

/// Regression: with `block` backpressure and no consumer, writers wait for
/// space; removing the stream and shutting down must not deadlock.
#[test]
fn block_backpressure_does_not_deadlock_on_stop() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let Some(ff) = ffmpeg() else {
        eprintln!("skipped: ffmpeg not found");
        return;
    };
    let path = clip(
        &ff,
        "block.mkv",
        "160x120",
        20,
        &["-c:v", "mjpeg", "-q:v", "5", "-pix_fmt", "yuvj420p"],
    )
    .expect("mjpeg encoder");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut out = out_cfg(32, 32);
        out.queue_depth = 1;
        let e =
            engine(file_url(&path), PlaybackConfig { realtime: false, looped: true }, None, out);
        // Nobody consumes: the output fills up and the writer blocks.
        std::thread::sleep(Duration::from_millis(800));
        e.remove_stream("f").unwrap();
        let mut s = StreamConfig::new("g", file_url(&path), &["o"]);
        s.playback = PlaybackConfig { realtime: false, looped: true };
        e.add_stream(s).unwrap();
        std::thread::sleep(Duration::from_millis(500));
        e.shutdown();
        tx.send(()).unwrap();
    });
    rx.recv_timeout(Duration::from_secs(20)).expect("remove_stream/shutdown deadlocked");
}
