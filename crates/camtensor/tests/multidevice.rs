//! Multi-device engines: placement, sharded outputs and per-device receive.
//!
//! Uses `cuda:0` + `cpu` as two devices (works on any machine with one GPU;
//! the logic is the same as with several GPUs). Skipped without CUDA/FFmpeg.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use camtensor::{
    Backpressure, DType, Device, Engine, EngineConfig, OutputConfig, PlaybackConfig,
    SnapshotOptions, StreamConfig, TensorSpec,
};

fn setup() -> Option<PathBuf> {
    if !camtensor::capabilities().cuda {
        eprintln!("skipped: CUDA not available");
        return None;
    }
    if std::env::var_os("CAMTENSOR_FFMPEG_DIR").is_none() {
        let dev = r"D:\github\portfolio\.devdeps\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1\bin";
        if Path::new(dev).is_dir() {
            // SAFETY: set before any engine thread starts.
            unsafe { std::env::set_var("CAMTENSOR_FFMPEG_DIR", dev) };
        }
    }
    let ff = [
        std::env::var("CAMTENSOR_FFMPEG").ok().map(PathBuf::from),
        Some(PathBuf::from(r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe")),
        Some(PathBuf::from("/usr/bin/ffmpeg")),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.exists())?;
    let dir = std::env::temp_dir().join(format!("camtensor-multi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let clip = dir.join("clip.mkv");
    let ok = Command::new(ff)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
        .arg("testsrc2=size=320x240:rate=25")
        .args(["-frames:v", "30", "-c:v", "mjpeg", "-q:v", "5", "-pix_fmt", "yuvj420p"])
        .arg(&clip)
        .status()
        .ok()?
        .success();
    ok.then_some(clip)
}

fn engine(clip: &Path, n: usize, pinned: &[(usize, Device)]) -> Engine {
    let mut cfg =
        EngineConfig { devices: vec![Device::Cuda(0), Device::Cpu], ..Default::default() };
    cfg.outputs.insert(
        "o".into(),
        OutputConfig {
            tensor: TensorSpec { width: 64, height: 64, dtype: DType::F16, ..Default::default() },
            keep_frame: true,
            backpressure: Backpressure::Block,
            queue_depth: 8,
            ..Default::default()
        },
    );
    for i in 0..n {
        let mut s = StreamConfig::new(format!("s{i}"), format!("file:{}", clip.display()), &["o"]);
        s.playback = PlaybackConfig { realtime: true, looped: true };
        s.device = pinned.iter().find(|(k, _)| *k == i).map(|(_, d)| *d);
        cfg.streams.push(s);
    }
    Engine::new(cfg).unwrap()
}

#[test]
fn streams_are_spread_and_batches_stay_on_their_device() {
    let Some(clip) = setup() else { return };
    let e = engine(&clip, 4, &[]);
    assert_eq!(e.devices(), vec![Device::Cuda(0), Device::Cpu]);

    // Placement: 4 streams over 2 devices -> 2 each.
    let st = e.stats();
    let mut per: HashMap<String, usize> = HashMap::new();
    for s in &st.streams {
        *per.entry(s.device.clone()).or_default() += 1;
    }
    assert_eq!(per.get("cuda:0"), Some(&2), "{per:?}");
    assert_eq!(per.get("cpu"), Some(&2), "{per:?}");
    let on_gpu: HashSet<String> =
        st.streams.iter().filter(|s| s.device == "cuda:0").map(|s| s.id.clone()).collect();

    // Per-device receive: only that device's streams, tensors on that device.
    for (device, expect_gpu) in [(Device::Cuda(0), true), (Device::Cpu, false)] {
        for _ in 0..5 {
            let b = e.recv_on("o", device, Some(Duration::from_secs(10))).unwrap();
            assert_eq!(b.tensor().device(), device);
            for it in b.items().iter().filter(|i| i.valid) {
                assert_eq!(
                    on_gpu.contains(&*it.meta.stream_id),
                    expect_gpu,
                    "{}",
                    it.meta.stream_id
                );
            }
        }
    }

    // Any-device receive sees both; crops/snapshots run on the frame's device.
    let mut seen = HashSet::new();
    let t0 = Instant::now();
    while seen.len() < 2 && t0.elapsed() < Duration::from_secs(10) {
        let b = e.recv("o", Some(Duration::from_secs(5))).unwrap();
        seen.insert(b.tensor().device());
        let item = &b.items()[0];
        let frame = item.frame.as_ref().unwrap();
        let jpeg = e.snapshot(frame, &SnapshotOptions::default()).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
        let crop = e
            .crop(
                frame,
                &[camtensor::Roi::Box { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }],
                &TensorSpec { width: 32, height: 32, ..Default::default() },
                None,
            )
            .unwrap();
        assert_eq!(crop.device(), b.tensor().device(), "crop stays on the frame's device");
    }
    assert_eq!(seen.len(), 2, "batches from both devices: {seen:?}");
    e.shutdown();
}

#[test]
fn pinning_and_load_based_placement() {
    let Some(clip) = setup() else { return };
    // s0 and s1 pinned to cuda:0; the next two go to the less loaded cpu.
    let e = engine(&clip, 4, &[(0, Device::Cuda(0)), (1, Device::Cuda(0))]);
    let st = e.stats();
    let dev = |id: &str| st.streams.iter().find(|s| s.id == id).unwrap().device.clone();
    assert_eq!(dev("s0"), "cuda:0");
    assert_eq!(dev("s1"), "cuda:0");
    assert_eq!(dev("s2"), "cpu");
    assert_eq!(dev("s3"), "cpu");

    // Pinning to a device the engine does not have is a configuration error.
    let mut bad = StreamConfig::new("x", format!("file:{}", clip.display()), &["o"]);
    bad.device = Some(Device::Cuda(7));
    assert!(matches!(e.add_stream(bad), Err(camtensor::Error::Config(_))));
    // Unknown shard.
    assert!(e.recv_on("o", Device::Cuda(7), Some(Duration::from_millis(10))).is_err());

    // Removing streams releases load: a new stream goes back to cuda:0.
    e.remove_stream("s0").unwrap();
    e.remove_stream("s1").unwrap();
    let mut s = StreamConfig::new("s4", format!("file:{}", clip.display()), &["o"]);
    s.playback = PlaybackConfig { realtime: true, looped: true };
    e.add_stream(s).unwrap();
    let st = e.stats();
    assert_eq!(st.streams.iter().find(|s| s.id == "s4").unwrap().device, "cuda:0");
    e.shutdown();
}
