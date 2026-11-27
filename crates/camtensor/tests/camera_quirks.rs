//! Real-camera behaviour emulated with MediaMTX, FFmpeg and a hostile TCP
//! proxy: digest authentication (right and wrong password), G.711 audio
//! next to video, camera reboot with a resolution change, a lossy network
//! (corrupted RTP payloads and dropped connections), and a slow camera with
//! a long GOP.
//!
//! Scenarios run one at a time (global lock) and use small, low-fps streams
//! to keep CPU usage low. Skipped when MediaMTX/FFmpeg are not available.

#[path = "../../camtensor-ingest/tests/common/mod.rs"]
mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use camtensor::{DType, Device, Engine, EngineConfig, OutputConfig, StreamConfig, TensorSpec};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

fn ffmpeg_dir() {
    if std::env::var_os("CAMTENSOR_FFMPEG_DIR").is_none() {
        let dev = r"D:\github\portfolio\.devdeps\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1\bin";
        if Path::new(dev).is_dir() {
            // SAFETY: set before any engine thread starts.
            unsafe { std::env::set_var("CAMTENSOR_FFMPEG_DIR", dev) };
        }
    }
}

/// MediaMTX with RTSP digest authentication: anyone may publish, only
/// `viewer:secret` may read.
struct AuthMtx {
    proc: common::Proc,
    dir: PathBuf,
    port: u16,
}

impl AuthMtx {
    fn start(exe: &Path, port: u16) -> Self {
        let dir =
            std::env::temp_dir().join(format!("camtensor-quirks-{}-{port}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let yml = format!(
            "logLevel: warn
api: no
rtsp: yes
rtspTransports: [tcp]
rtspAddress: 127.0.0.1:{port}
rtspAuthMethods: [digest]
rtmp: no
hls: no
webrtc: no
srt: no
moq: no
authInternalUsers:
  - user: any
    pass:
    ips: []
    permissions:
      - action: publish
  - user: viewer
    pass: secret
    ips: []
    permissions:
      - action: read
paths:
  all_others:
"
        );
        std::fs::write(dir.join("mediamtx.yml"), yml).unwrap();
        let child = Command::new(exe)
            .arg(dir.join("mediamtx.yml"))
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start mediamtx");
        let deadline = Instant::now() + Duration::from_secs(10);
        while TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(Instant::now() < deadline, "mediamtx did not start");
            std::thread::sleep(Duration::from_millis(50));
        }
        Self { proc: common::Proc(Some(child)), dir, port }
    }

    fn url(&self, path: &str) -> String {
        format!("rtsp://127.0.0.1:{}/{path}", self.port)
    }

    fn viewer_url(&self, path: &str) -> String {
        format!("rtsp://viewer:secret@127.0.0.1:{}/{path}", self.port)
    }
}

impl Drop for AuthMtx {
    fn drop(&mut self) {
        self.proc.kill();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A camera-like publisher: H.264 Main, no B-frames, small and slow.
fn camera(ffmpeg: &Path, url: &str, size: &str, fps: u32, gop: u32, audio: bool) -> common::Proc {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin", "-re", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={size}:rate={fps}"));
    if audio {
        cmd.args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000"]);
    }
    cmd.args([
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "main",
    ])
    .args(["-bf", "0", "-g"])
    .arg(gop.to_string())
    .args(["-pix_fmt", "yuv420p"]);
    if audio {
        // G.711 A-law, the most common camera audio codec.
        cmd.args(["-c:a", "pcm_alaw", "-ar", "8000", "-ac", "1"]);
    }
    cmd.args(["-f", "rtsp", "-rtsp_transport", "tcp", url])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    common::Proc(Some(cmd.spawn().expect("start ffmpeg")))
}

fn engine_for(url: &str) -> Engine {
    let mut cfg = EngineConfig { device: Device::Auto, ..Default::default() };
    cfg.outputs.insert(
        "o".into(),
        OutputConfig {
            tensor: TensorSpec { width: 96, height: 96, dtype: DType::U8, ..Default::default() },
            ..Default::default()
        },
    );
    let mut s = StreamConfig::new("cam", url, &["o"]);
    s.max_reconnect_ms = 1000;
    cfg.streams.push(s);
    Engine::new(cfg).expect("engine")
}

/// Receives images until `pred` holds for one of them or `timeout` passes.
fn wait_image(
    engine: &Engine,
    timeout: Duration,
    pred: impl Fn(&camtensor::BatchItem) -> bool,
) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(b) = engine.recv("o", Some(Duration::from_millis(200)))
            && b.items().iter().any(|i| i.valid && pred(i))
        {
            return true;
        }
    }
    false
}

fn count_images(engine: &Engine, d: Duration) -> usize {
    let deadline = Instant::now() + d;
    let mut n = 0;
    while Instant::now() < deadline {
        if let Ok(b) = engine.recv("o", Some(Duration::from_millis(200))) {
            n += b.valid_count();
        }
    }
    n
}

fn wait_publisher(url: &str) {
    // The path becomes readable once the publisher's first frames arrive.
    std::thread::sleep(Duration::from_millis(1500));
    let _ = url;
}

#[test]
fn digest_auth_right_and_wrong_password() {
    let _g = serial();
    let Some((mtx_exe, ffmpeg)) = common::tools("quirks-auth") else { return };
    ffmpeg_dir();
    let mtx = AuthMtx::start(&mtx_exe, 18700);
    let _cam = camera(&ffmpeg, &mtx.url("cam"), "640x360", 15, 15, false);
    wait_publisher(&mtx.url("cam"));

    let ok = engine_for(&mtx.viewer_url("cam"));
    assert!(wait_image(&ok, Duration::from_secs(15), |_| true), "{:#?}", ok.stats());
    ok.shutdown();

    // Credentials given separately from the URL.
    let mut cfg = EngineConfig::default();
    cfg.outputs.insert(
        "o".into(),
        OutputConfig {
            tensor: TensorSpec { width: 64, height: 64, dtype: DType::U8, ..Default::default() },
            ..Default::default()
        },
    );
    let mut s = StreamConfig::new("cam", mtx.url("cam"), &["o"]);
    s.username = Some("viewer".into());
    s.password = Some("secret".into());
    cfg.streams.push(s);
    let sep = Engine::new(cfg).unwrap();
    assert!(wait_image(&sep, Duration::from_secs(15), |_| true), "{:#?}", sep.stats());
    sep.shutdown();

    let bad = engine_for(&format!("rtsp://viewer:wrong@127.0.0.1:{}/cam", mtx.port));
    assert_eq!(count_images(&bad, Duration::from_secs(3)), 0);
    // Under load the first attempts may take a while: wait for the condition.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut st = bad.stats().streams[0].clone();
    while (st.reconnects < 1 || st.last_error.is_none()) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
        st = bad.stats().streams[0].clone();
    }
    assert_eq!(count_images(&bad, Duration::from_millis(500)), 0);
    assert!(!st.connected);
    assert!(st.reconnects >= 1, "keeps retrying: {st:?}");
    let err = st.last_error.clone().unwrap_or_default().to_lowercase();
    assert!(err.contains("401") || err.contains("auth"), "error explains the failure: {err}");
    bad.shutdown();
}

#[test]
fn audio_track_next_to_video() {
    let _g = serial();
    let Some((mtx_exe, ffmpeg)) = common::tools("quirks-audio") else { return };
    ffmpeg_dir();
    let mtx = AuthMtx::start(&mtx_exe, 18710);
    let _cam = camera(&ffmpeg, &mtx.url("cam"), "640x360", 15, 15, true);
    wait_publisher(&mtx.url("cam"));
    let e = engine_for(&mtx.viewer_url("cam"));
    assert!(count_images(&e, Duration::from_secs(6)) >= 15, "{:#?}", e.stats());
    assert_eq!(e.stats().streams[0].codec.as_deref(), Some("h264"));
    e.shutdown();
}

#[test]
fn camera_reboot_with_resolution_change() {
    let _g = serial();
    let Some((mtx_exe, ffmpeg)) = common::tools("quirks-reboot") else { return };
    ffmpeg_dir();
    let mtx = AuthMtx::start(&mtx_exe, 18720);
    let mut cam = camera(&ffmpeg, &mtx.url("cam"), "640x360", 15, 15, false);
    wait_publisher(&mtx.url("cam"));
    let e = engine_for(&mtx.viewer_url("cam"));
    assert!(wait_image(&e, Duration::from_secs(15), |i| i.meta.src_width == 640));

    // "Reboot": the camera disappears for a few seconds and comes back with
    // a different resolution (a common reconfiguration).
    cam.kill();
    std::thread::sleep(Duration::from_secs(3));
    let _cam2 = camera(&ffmpeg, &mtx.url("cam"), "1280x720", 15, 15, false);
    assert!(
        wait_image(&e, Duration::from_secs(25), |i| i.meta.src_width == 1280
            && i.meta.src_height == 720),
        "{:#?}",
        e.stats()
    );
    assert!(e.stats().streams[0].reconnects >= 1);
    e.shutdown();
}

/// TCP proxy that corrupts a fraction of the RTP payloads (interleaved
/// `$` frames) going to the client and drops every connection periodically.
struct HostileProxy {
    port: u16,
    stop: Arc<AtomicBool>,
    corrupted: Arc<AtomicU64>,
    cuts: Arc<AtomicU64>,
}

impl HostileProxy {
    fn start(port: u16, upstream: u16, corrupt_one_in: u64, cut_every: Duration) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let corrupted = Arc::new(AtomicU64::new(0));
        let cuts = Arc::new(AtomicU64::new(0));
        let (s, c, k) = (stop.clone(), corrupted.clone(), cuts.clone());
        std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((client, _)) => {
                        let _ = client.set_nonblocking(false);
                        if let Ok(server) = TcpStream::connect(("127.0.0.1", upstream)) {
                            relay(
                                client,
                                server,
                                corrupt_one_in,
                                cut_every,
                                s.clone(),
                                c.clone(),
                                k.clone(),
                            );
                        }
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(20)),
                }
            }
        });
        Self { port, stop, corrupted, cuts }
    }
}

impl Drop for HostileProxy {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn relay(
    client: TcpStream,
    server: TcpStream,
    corrupt_one_in: u64,
    cut_every: Duration,
    stop: Arc<AtomicBool>,
    corrupted: Arc<AtomicU64>,
    cuts: Arc<AtomicU64>,
) {
    let (mut c_rd, mut s_wr) = (client.try_clone().unwrap(), server.try_clone().unwrap());
    // Client -> server: plain copy.
    std::thread::spawn(move || {
        let _ = std::io::copy(&mut c_rd, &mut s_wr);
    });
    // Server -> client: parse interleaved frames and RTSP messages.
    let killer = (client.try_clone().unwrap(), server.try_clone().unwrap());
    let started = Instant::now();
    std::thread::spawn(move || {
        let mut rd = BufReader::new(server);
        let mut wr = client;
        let mut seq: u64 = 0;
        loop {
            if stop.load(Ordering::Relaxed) || started.elapsed() >= cut_every {
                cuts.fetch_add(1, Ordering::Relaxed);
                let _ = killer.0.shutdown(std::net::Shutdown::Both);
                let _ = killer.1.shutdown(std::net::Shutdown::Both);
                return;
            }
            let mut first = [0u8; 1];
            if rd.read_exact(&mut first).is_err() {
                return;
            }
            if first[0] == b'$' {
                let mut hdr = [0u8; 3];
                if rd.read_exact(&mut hdr).is_err() {
                    return;
                }
                let len = u16::from_be_bytes([hdr[1], hdr[2]]) as usize;
                let mut payload = vec![0u8; len];
                if rd.read_exact(&mut payload).is_err() {
                    return;
                }
                seq += 1;
                // Even channels carry RTP; flip a few payload bytes after the
                // 12-byte RTP header (+ 2 bytes of the H.264 FU header).
                if hdr[0] % 2 == 0 && len > 40 && seq.is_multiple_of(corrupt_one_in) {
                    for k in 0..4 {
                        let i = 14 + ((seq as usize * 31 + k * 97) % (len - 14));
                        payload[i] ^= 0x5a;
                    }
                    corrupted.fetch_add(1, Ordering::Relaxed);
                }
                if wr
                    .write_all(b"$")
                    .and_then(|_| wr.write_all(&hdr))
                    .and_then(|_| wr.write_all(&payload))
                    .is_err()
                {
                    return;
                }
            } else {
                // RTSP response: headers until the blank line, then the body.
                let mut msg = vec![first[0]];
                let mut line = String::new();
                let mut content_len = 0usize;
                loop {
                    line.clear();
                    if rd.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    msg.extend_from_slice(line.as_bytes());
                    let lower = line.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("content-length:") {
                        content_len = v.trim().parse().unwrap_or(0);
                    }
                    if line == "\r\n" || line == "\n" {
                        break;
                    }
                }
                let mut body = vec![0u8; content_len];
                if rd.read_exact(&mut body).is_err() {
                    return;
                }
                msg.extend_from_slice(&body);
                if wr.write_all(&msg).is_err() {
                    return;
                }
            }
        }
    });
}

#[test]
fn lossy_network_corruption_and_disconnects() {
    let _g = serial();
    let Some((mtx_exe, ffmpeg)) = common::tools("quirks-lossy") else { return };
    ffmpeg_dir();
    let mtx = AuthMtx::start(&mtx_exe, 18730);
    let _cam = camera(&ffmpeg, &mtx.url("cam"), "640x360", 15, 15, false);
    wait_publisher(&mtx.url("cam"));
    // 1 in 40 RTP packets corrupted, every connection dropped after 5 s.
    let proxy = HostileProxy::start(18738, mtx.port, 40, Duration::from_secs(5));
    let e = engine_for(&format!("rtsp://viewer:secret@127.0.0.1:{}/cam", proxy.port));
    let images = count_images(&e, Duration::from_secs(18));
    let st = e.stats().streams[0].clone();
    eprintln!(
        "lossy: images={images} corrupted={} cuts={} reconnects={} decode_errors={} last_error={:?}",
        proxy.corrupted.load(Ordering::Relaxed),
        proxy.cuts.load(Ordering::Relaxed),
        st.reconnects,
        st.decode_errors,
        st.last_error
    );
    assert!(proxy.corrupted.load(Ordering::Relaxed) > 0, "corruption was injected");
    assert!(proxy.cuts.load(Ordering::Relaxed) >= 2, "connections were dropped");
    assert!(st.reconnects >= 2, "the engine reconnected: {st:?}");
    // Most of the 18 s x 15 fps survive despite corruption and outages.
    assert!(images >= 90, "images delivered through a hostile network: {images}");
    e.shutdown();
}

#[test]
fn slow_camera_with_long_gop() {
    let _g = serial();
    let Some((mtx_exe, ffmpeg)) = common::tools("quirks-slow") else { return };
    ffmpeg_dir();
    let mtx = AuthMtx::start(&mtx_exe, 18740);
    // 5 fps with a keyframe every 10 s (energy-saving / low-bandwidth cameras).
    let _cam = camera(&ffmpeg, &mtx.url("cam"), "640x360", 5, 50, false);
    wait_publisher(&mtx.url("cam"));
    let t0 = Instant::now();
    let e = engine_for(&mtx.viewer_url("cam"));
    assert!(wait_image(&e, Duration::from_secs(14), |_| true), "first image within one GOP");
    eprintln!("slow camera: first image after {:?}", t0.elapsed());
    let n = count_images(&e, Duration::from_secs(4));
    assert!((14..=26).contains(&n), "~5 fps after the first keyframe: {n}");
    e.shutdown();
}
