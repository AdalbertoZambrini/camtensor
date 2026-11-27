//! Test harness: MediaMTX on private ports plus FFmpeg publishers, all
//! killed on drop (also when a test panics).

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_MEDIAMTX: &str = r"C:\mediamtx-ffmpeg\mediamtx\mediamtx.exe";
const DEFAULT_FFMPEG: &str = r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe";

fn find_exe(env: &str, default: &str, name: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let p = PathBuf::from(default);
    if p.is_file() {
        return Some(p);
    }
    // Fall back on PATH.
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_owned() };
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(&exe)).find(|p| p.is_file()))
}

/// Paths of the external tools, or `None` (with a printed message) to skip.
pub fn tools(test: &str) -> Option<(PathBuf, PathBuf)> {
    let mtx = find_exe("CAMTENSOR_MEDIAMTX", DEFAULT_MEDIAMTX, "mediamtx");
    let ff = find_exe("CAMTENSOR_FFMPEG", DEFAULT_FFMPEG, "ffmpeg");
    match (mtx, ff) {
        (Some(m), Some(f)) => Some((m, f)),
        _ => {
            eprintln!(
                "skipping {test}: mediamtx/ffmpeg not found (set CAMTENSOR_MEDIAMTX / CAMTENSOR_FFMPEG)"
            );
            None
        }
    }
}

/// A child process killed and reaped on drop.
pub struct Proc(pub Option<Child>);

impl Proc {
    pub fn kill(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        self.kill();
    }
}

/// A MediaMTX instance on test ports with its own config directory.
pub struct MediaMtx {
    proc: Proc,
    dir: PathBuf,
    pub rtsp_port: u16,
    pub api_port: u16,
}

impl MediaMtx {
    /// Starts MediaMTX with RTSP on `base`, RTP/RTCP on `base + 2/3` and the
    /// API on `base + 4`.
    pub fn start(exe: &Path, base: u16) -> Self {
        assert!(base.is_multiple_of(2), "base port must be even");
        let dir =
            std::env::temp_dir().join(format!("camtensor-test-{}-{base}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (rtp, rtcp, api) = (base + 2, base + 3, base + 4);
        let yml = format!(
            "logLevel: info
api: yes
apiAddress: 127.0.0.1:{api}
rtsp: yes
rtspTransports: [udp, tcp]
rtspAddress: 127.0.0.1:{base}
rtpAddress: 127.0.0.1:{rtp}
rtcpAddress: 127.0.0.1:{rtcp}
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
      - action: read
      - action: api
paths:
  all_others:
"
        );
        std::fs::write(dir.join("mediamtx.yml"), yml).unwrap();
        let log = std::fs::File::create(dir.join("mediamtx.log")).unwrap();
        let child = Command::new(exe)
            .arg(dir.join("mediamtx.yml"))
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .expect("start mediamtx");
        let m = Self { proc: Proc(Some(child)), dir, rtsp_port: base, api_port: api };
        let deadline = Instant::now() + Duration::from_secs(10);
        while TcpStream::connect(("127.0.0.1", base)).is_err() {
            assert!(Instant::now() < deadline, "mediamtx did not start; log: {}", m.log());
            std::thread::sleep(Duration::from_millis(50));
        }
        m
    }

    pub fn url(&self, path: &str) -> String {
        format!("rtsp://127.0.0.1:{}/{path}", self.rtsp_port)
    }

    pub fn log(&self) -> String {
        std::fs::read_to_string(self.dir.join("mediamtx.log")).unwrap_or_default()
    }

    /// Waits until the path has a ready publisher (MediaMTX API).
    pub fn wait_ready(&self, path: &str, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            if self
                .api_get(&format!("/v3/paths/get/{path}"))
                .is_some_and(|b| b.contains("\"ready\":true"))
            {
                return;
            }
            assert!(Instant::now() < deadline, "path {path} not ready; log: {}", self.log());
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn api_get(&self, path: &str) -> Option<String> {
        let mut s = TcpStream::connect(("127.0.0.1", self.api_port)).ok()?;
        s.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
        write!(s, "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").ok()?;
        let mut body = String::new();
        s.read_to_string(&mut body).ok()?;
        Some(body)
    }
}

impl Drop for MediaMtx {
    fn drop(&mut self) {
        self.proc.kill();
        if std::thread::panicking() {
            eprintln!("--- mediamtx log ---\n{}", self.log());
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Starts an endless FFmpeg `testsrc2` publisher with the given encoder args.
pub fn publish(ffmpeg: &Path, url: &str, size: &str, encoder: &[&str]) -> Proc {
    let src = format!("testsrc2=size={size}:rate=25");
    let child = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-re", "-f", "lavfi", "-i", &src])
        .args(encoder)
        .args(["-f", "rtsp", "-rtsp_transport", "tcp", url])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start ffmpeg");
    Proc(Some(child))
}

pub const H264: &[&str] = &[
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-tune",
    "zerolatency",
    "-g",
    "25",
    "-pix_fmt",
    "yuv420p",
];
pub const H265: &[&str] = &[
    "-c:v",
    "libx265",
    "-preset",
    "ultrafast",
    "-tune",
    "zerolatency",
    "-g",
    "25",
    "-pix_fmt",
    "yuv420p",
    "-x265-params",
    "log-level=error",
];
/// RFC 2435 (RTP/JPEG) needs the standard Huffman tables, and MediaMTX
/// rejects frames that share one quantisation table between luma and chroma
/// ("received wrong fragment"), hence `-force_duplicated_matrix 1`.
pub const MJPEG: &[&str] = &[
    "-c:v",
    "mjpeg",
    "-q:v",
    "5",
    "-huffman",
    "default",
    "-force_duplicated_matrix",
    "1",
    "-pix_fmt",
    "yuvj420p",
];
