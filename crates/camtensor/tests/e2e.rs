//! End-to-end tests: MediaMTX + FFmpeg publishers → engine → tensors.
//!
//! Skipped (with a message) when MediaMTX or FFmpeg are not available. Set
//! `CAMTENSOR_MEDIAMTX` / `CAMTENSOR_FFMPEG` to override the tool paths and
//! `CAMTENSOR_FFMPEG_DIR` to point at shared libavcodec for the CPU decoder.

#[path = "../../camtensor-ingest/tests/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use camtensor::{
    BatchConfig, DType, Device, Engine, EngineConfig, GatingConfig, GatingMode, Layout,
    MediaMtxConfig, Normalize, OutputConfig, ResizeMode, Roi, StreamConfig, TensorSpec,
};

fn ffmpeg_dir() {
    if std::env::var_os("CAMTENSOR_FFMPEG_DIR").is_none() {
        let dev = r"D:\github\portfolio\.devdeps\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1\bin";
        if std::path::Path::new(dev).is_dir() {
            // SAFETY: set once at test start before any thread reads it.
            unsafe { std::env::set_var("CAMTENSOR_FFMPEG_DIR", dev) };
        }
    }
}

fn outputs() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    cfg.outputs.insert(
        "det".into(),
        OutputConfig {
            tensor: TensorSpec {
                width: 320,
                height: 320,
                dtype: DType::F32,
                resize: ResizeMode::Letterbox,
                normalize: Normalize::UNIT,
                ..Default::default()
            },
            batch: BatchConfig { max_size: 4, max_wait_ms: 40, pad_to_max: false },
            keep_frame: true,
            ..Default::default()
        },
    );
    cfg.outputs.insert(
        "thumb".into(),
        OutputConfig {
            tensor: TensorSpec {
                width: 64,
                height: 64,
                dtype: DType::U8,
                layout: Layout::Nhwc,
                resize: ResizeMode::Stretch,
                ..Default::default()
            },
            max_fps: Some(5.0),
            ..Default::default()
        },
    );
    cfg
}

/// Collects batches until every stream delivered `n` images on `output`.
fn collect(engine: &Engine, output: &str, streams: &[&str], n: usize) -> Vec<camtensor::Batch> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    while streams.iter().any(|s| seen.get(*s).copied().unwrap_or(0) < n) {
        assert!(
            Instant::now() < deadline,
            "timed out on {output}: {seen:?}\n{:#?}",
            engine.stats()
        );
        if let Ok(b) = engine.recv(output, Some(Duration::from_millis(200))) {
            for it in b.items().iter().filter(|i| i.valid) {
                *seen.entry(it.meta.stream_id.to_string()).or_default() += 1;
            }
            // Holding every batch would exhaust the output's buffer pool (and
            // make the engine drop frames, by design): keep only a few.
            if out.len() < 3 {
                out.push(b);
            }
        }
    }
    out
}

fn host(engine: &Engine, t: &camtensor::Tensor) -> Vec<u8> {
    engine.to_host_vec(t).expect("download")
}

fn run_pipeline(device: Device) {
    let Some((mtx_exe, ffmpeg)) = common::tools("e2e") else { return };
    ffmpeg_dir();
    let base = if device == Device::Cpu { 18600 } else { 18620 };
    let mtx = common::MediaMtx::start(&mtx_exe, base);
    let _p1 = common::publish(&ffmpeg, &mtx.url("h264"), "1280x720", common::H264);
    let _p2 = common::publish(&ffmpeg, &mtx.url("h265"), "1280x720", common::H265);
    let _p3 = common::publish(&ffmpeg, &mtx.url("mjpeg"), "640x360", common::MJPEG);
    for p in ["h264", "h265", "mjpeg"] {
        mtx.wait_ready(p, Duration::from_secs(15));
    }

    let mut cfg = outputs();
    cfg.device = device;
    for p in ["h264", "h265", "mjpeg"] {
        cfg.streams.push(StreamConfig::new(p, mtx.url(p), &["det", "thumb"]));
    }
    let engine = match Engine::new(cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipped: cannot create engine on {device}: {e}");
            return;
        }
    };
    let streams = ["h264", "h265", "mjpeg"];
    let batches = collect(&engine, "det", &streams, 8);

    // Shapes, metadata and letterbox content.
    let mut checked = 0;
    for b in &batches {
        let t = b.tensor();
        assert_eq!(&t.shape()[1..], &[3, 320, 320]);
        assert_eq!(t.shape()[0], b.len());
        assert!(b.len() <= 4);
        let data = host(&engine, t);
        let floats: &[f32] = bytemuck_f32(&data);
        for (i, it) in b.items().iter().enumerate() {
            let m = &it.meta;
            let (sw, sh) = (m.src_width as f32, m.src_height as f32);
            assert!(sw > 0.0 && sh > 0.0);
            // Letterbox: x scale = 320/src_w, centred vertically.
            let s = 320.0 / sw;
            assert!((m.transform.0[0] - s).abs() < 1e-4);
            let pad = (320.0 - sh * s) / 2.0;
            assert!((m.transform.0[5] - pad).abs() < 1e-3);
            let img = &floats[i * 3 * 320 * 320..(i + 1) * 3 * 320 * 320];
            // Top-left pixel is padding (114/255), centre is image content in [0, 1].
            assert!((img[0] - 114.0 / 255.0).abs() < 1e-3, "pad value {}", img[0]);
            assert!(img.iter().all(|v| (0.0..=1.0).contains(v)));
            let (x, y) = m.to_source(160.0, 160.0);
            assert!((x - sw / 2.0).abs() < 1.0 && (y - sh / 2.0).abs() < 1.0);
            assert!(it.frame.is_some(), "keep_frame");
            checked += 1;
        }
    }
    assert!(checked >= 3);

    // Second output: u8 NHWC thumbnails, rate-capped to ~5 fps.
    let thumbs = collect(&engine, "thumb", &streams, 3);
    let t = thumbs[0].tensor();
    assert_eq!(t.shape(), [1, 64, 64, 3]);
    assert_eq!(t.dtype(), DType::U8);

    // Crops from a kept frame: a box and an ArcFace-aligned landmark crop.
    let item = batches.iter().flat_map(|b| b.items()).find(|i| i.valid).unwrap();
    let frame = item.frame.as_ref().unwrap();
    let crop_spec = TensorSpec {
        width: 112,
        height: 112,
        resize: ResizeMode::Stretch,
        normalize: Normalize::SYMMETRIC,
        ..Default::default()
    };
    let rois = [
        Roi::Box { x: 10.0, y: 10.0, w: 200.0, h: 200.0 },
        Roi::Landmarks {
            points: [
                [300.0, 200.0],
                [360.0, 198.0],
                [330.0, 240.0],
                [305.0, 280.0],
                [355.0, 278.0],
            ],
        },
    ];
    let crops = engine.crop(frame, &rois, &crop_spec, None).unwrap();
    assert_eq!(crops.shape(), [2, 3, 112, 112]);
    let c = host(&engine, &crops);
    assert!(bytemuck_f32(&c).iter().all(|v| (-1.0..=1.0).contains(v)));

    let st = engine.stats();
    for s in &st.streams {
        assert!(s.connected, "{s:?}");
        assert!(s.frames > 0);
        eprintln!(
            "{} {:?} decoder={:?} latency={:.2}ms",
            s.id, s.codec, s.decoder, s.avg_latency_ms
        );
    }
    assert!(st.to_prometheus().contains("camtensor_stream_frames_total"));
    engine.shutdown();
}

fn bytemuck_f32(b: &[u8]) -> &[f32] {
    assert_eq!(b.as_ptr() as usize % 4, 0);
    // SAFETY: aligned (checked), length is a multiple of 4 for f32 tensors.
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}

#[test]
fn pipeline_cpu() {
    run_pipeline(Device::Cpu);
}

#[test]
fn pipeline_cuda() {
    if !camtensor::capabilities().cuda {
        eprintln!("skipped: CUDA not available");
        return;
    }
    run_pipeline(Device::Cuda(0));
}

#[test]
fn mediamtx_discovery_and_gating() {
    let Some((mtx_exe, ffmpeg)) = common::tools("discovery") else { return };
    ffmpeg_dir();
    let mtx = common::MediaMtx::start(&mtx_exe, 18640);
    let _p1 = common::publish(&ffmpeg, &mtx.url("cam1"), "640x360", common::H264);
    let _p2 = common::publish(&ffmpeg, &mtx.url("cam2"), "640x360", common::H264);
    let _p3 = common::publish(&ffmpeg, &mtx.url("door"), "640x360", common::H264);
    for p in ["cam1", "cam2", "door"] {
        mtx.wait_ready(p, Duration::from_secs(15));
    }
    let mut cfg = outputs();
    cfg.device = Device::Auto;
    cfg.mediamtx = Some(MediaMtxConfig {
        api_url: format!("http://127.0.0.1:{}", mtx.api_port),
        rtsp_url: format!("rtsp://127.0.0.1:{}", mtx.rtsp_port),
        paths: vec!["cam*".into()],
        outputs: vec!["det".into()],
        gating: GatingConfig { mode: GatingMode::Keyframes, ..Default::default() },
        poll_interval_ms: 300,
        ..Default::default()
    });
    let engine = Engine::new(cfg).unwrap();
    collect(&engine, "det", &["cam1", "cam2"], 3);
    let mut ids = engine.streams();
    ids.sort();
    assert_eq!(ids, vec!["cam1", "cam2"], "only paths matching cam* are consumed");
    // Keyframes-only gating: testsrc2 at 25 fps with GOP 25 → ~1 decoded frame/s.
    std::thread::sleep(Duration::from_secs(3));
    let st = engine.stats();
    for s in &st.streams {
        assert!(s.skipped_packets > s.decoded_packets, "{s:?}");
    }
    engine.shutdown();
}
