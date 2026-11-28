//! Declarative configuration (YAML / JSON) for an engine.
//!
//! ```yaml
//! device: auto            # auto | cpu | cuda:N
//! outputs:
//!   detector:
//!     tensor: { width: 640, height: 640, dtype: f16, resize: letterbox, normalize: unit }
//!     batch: { max_size: 8, max_wait_ms: 10 }
//!     keep_frame: true
//! streams:
//!   - id: cam1
//!     url: rtsp://127.0.0.1:8554/cam1
//!     outputs: [detector]
//!     gating: { mode: auto }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::tensor::{Device, TensorSpec};
use crate::{Error, Result};

/// Top-level engine configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// Default device for decoding and output tensors.
    pub device: Device,
    /// Several devices (multi-GPU). Each stream is placed on one of them
    /// (least loaded by pixel rate, unless the stream pins `device`) and
    /// stays there; outputs get one shard per device. `auto` in the list
    /// expands to every CUDA device. Empty = just `device`.
    pub devices: Vec<Device>,
    /// Default decoder preference for all streams.
    pub decoder: DecoderPref,
    /// Named outputs. Streams reference outputs by name; all streams that
    /// share an output are batched together.
    pub outputs: BTreeMap<String, OutputConfig>,
    /// Statically configured streams.
    pub streams: Vec<StreamConfig>,
    /// Optional MediaMTX auto-discovery.
    pub mediamtx: Option<MediaMtxConfig>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device: Device::Auto,
            devices: Vec::new(),
            decoder: DecoderPref::Auto,
            outputs: BTreeMap::new(),
            streams: Vec::new(),
            mediamtx: None,
        }
    }
}

impl EngineConfig {
    /// Parses YAML (JSON is valid YAML too).
    pub fn from_yaml(s: &str) -> Result<Self> {
        let cfg: Self = serde_norway::from_str(s).map_err(Error::config)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reads and parses a YAML/JSON file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Self::from_yaml(&text)
    }

    /// Serialises to YAML.
    pub fn to_yaml(&self) -> String {
        serde_norway::to_string(self).unwrap_or_default()
    }

    /// Checks cross references and value ranges.
    pub fn validate(&self) -> Result<()> {
        for (name, out) in &self.outputs {
            validate_name(name, "output")?;
            out.validate().map_err(|e| Error::config(format!("output '{name}': {e}")))?;
        }
        let mut seen = std::collections::HashSet::new();
        for s in &self.streams {
            s.validate()?;
            if !seen.insert(s.id.as_str()) {
                return Err(Error::config(format!("duplicate stream id '{}'", s.id)));
            }
            for o in &s.outputs {
                if !self.outputs.contains_key(o) {
                    return Err(Error::config(format!(
                        "stream '{}' references unknown output '{o}'",
                        s.id
                    )));
                }
            }
        }
        if let Some(m) = &self.mediamtx {
            for o in &m.outputs {
                if !self.outputs.contains_key(o) {
                    return Err(Error::config(format!("mediamtx references unknown output '{o}'")));
                }
            }
        }
        Ok(())
    }
}

fn validate_name(name: &str, what: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c))
    {
        return Err(Error::config(format!(
            "invalid {what} name '{name}' (use [A-Za-z0-9-_./], max 128 chars)"
        )));
    }
    Ok(())
}

/// Which decoder implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecoderPref {
    /// NVDEC when the device is CUDA and the codec is supported, CPU otherwise.
    #[default]
    Auto,
    /// Force NVDEC (fails if unavailable).
    Nvdec,
    /// Force the CPU decoder.
    Cpu,
}

/// What to do when a consumer falls behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backpressure {
    /// Drop the oldest undelivered batch (bounded latency, default).
    #[default]
    DropOldest,
    /// Keep only the most recent batch.
    LatestOnly,
    /// Stall producers until the consumer catches up (no frame loss, latency
    /// grows; the RTSP socket may overflow on slow consumers).
    Block,
}

/// Batching policy of an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BatchConfig {
    /// Maximum images per batch.
    pub max_size: usize,
    /// Maximum time to wait for a batch to fill, measured from its first
    /// image. `0` emits as soon as an image is ready when `max_size` is 1,
    /// otherwise waits until full.
    pub max_wait_ms: u64,
    /// Always deliver `max_size` images (unused slots zero-filled and marked
    /// invalid). Useful for engines with a static batch dimension.
    pub pad_to_max: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self { max_size: 1, max_wait_ms: 0, pad_to_max: false }
    }
}

/// One named output: a tensor format plus delivery policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Image format.
    pub tensor: TensorSpec,
    /// Batching policy.
    pub batch: BatchConfig,
    /// Per-stream frame-rate cap for this output (`None` = every decoded frame).
    pub max_fps: Option<f64>,
    /// Keep a reference to the full-resolution decoded frame with each image
    /// so the consumer can request crops (`Batch::crop`).
    pub keep_frame: bool,
    /// Consumer lag policy.
    pub backpressure: Backpressure,
    /// Number of batches that may be in flight (delivered but not yet
    /// released by the consumer, plus queued).
    pub queue_depth: usize,
    /// Device for this output's tensors (defaults to the engine device).
    pub device: Option<Device>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            tensor: TensorSpec::default(),
            batch: BatchConfig::default(),
            max_fps: None,
            keep_frame: false,
            backpressure: Backpressure::DropOldest,
            queue_depth: 4,
            device: None,
        }
    }
}

impl OutputConfig {
    /// Validates value ranges.
    pub fn validate(&self) -> Result<()> {
        self.tensor.validate()?;
        if self.batch.max_size == 0 || self.batch.max_size > 1024 {
            return Err(Error::config("batch.max_size must be in 1..=1024"));
        }
        if self.queue_depth == 0 {
            return Err(Error::config("queue_depth must be >= 1"));
        }
        if let Some(f) = self.max_fps
            && (f.is_nan() || f <= 0.0)
        {
            return Err(Error::config("max_fps must be > 0"));
        }
        Ok(())
    }
}

/// RTSP transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    /// Interleaved over the RTSP TCP connection (robust, default).
    #[default]
    Tcp,
    /// RTP over UDP (lower overhead on clean networks).
    Udp,
}

/// Gating strategy: how much of the stream to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GatingMode {
    /// Decode every frame.
    Always,
    /// Decode keyframes only.
    Keyframes,
    /// Decode keyframes while the scene is static; decode everything while
    /// activity is detected from compressed-domain statistics.
    #[default]
    Auto,
}

/// Gating parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GatingConfig {
    /// Strategy.
    pub mode: GatingMode,
    /// Activity threshold: a non-key packet larger than `threshold` times
    /// the idle baseline counts as activity.
    pub threshold: f32,
    /// Consecutive active packets required to switch to full decoding.
    pub trigger_frames: u32,
    /// Number of recent non-key packets averaged before scoring (H.264/H.265).
    /// Smooths the size alternation of B/P frame patterns; `1` scores every
    /// packet on its own.
    pub window: u32,
    /// Keep decoding everything for this long after the last activity.
    pub hold_ms: u64,
    /// Packets used to learn the idle baseline before gating kicks in
    /// (everything is decoded during warm-up).
    pub warmup_frames: u32,
    /// For intra-only codecs (MJPEG), frame rate decoded while idle.
    pub idle_fps: f64,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            mode: GatingMode::Always,
            threshold: 1.6,
            trigger_frames: 2,
            window: 8,
            hold_ms: 2000,
            warmup_frames: 50,
            idle_fps: 1.0,
        }
    }
}

/// Region of interest in **normalised** frame coordinates (`0..=1`), so it
/// survives camera resolution changes. Only this part of the frame is fed to
/// the outputs; `FrameMeta::transform` still maps back to full-frame pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Region {
    /// Left edge (fraction of the width).
    pub x: f32,
    /// Top edge (fraction of the height).
    pub y: f32,
    /// Width (fraction of the width).
    pub w: f32,
    /// Height (fraction of the height).
    pub h: f32,
}

impl Region {
    /// Validates that the region is non-empty and inside the frame.
    pub fn validate(&self) -> Result<()> {
        let ok = [self.x, self.y, self.w, self.h].iter().all(|v| v.is_finite())
            && self.x >= 0.0
            && self.y >= 0.0
            && self.w > 0.0
            && self.h > 0.0
            && self.x + self.w <= 1.0 + 1e-6
            && self.y + self.h <= 1.0 + 1e-6;
        if ok {
            Ok(())
        } else {
            Err(Error::config("region must be normalised (0..=1), non-empty and inside the frame"))
        }
    }

    /// The region in pixels of a `width x height` frame: `(x, y, w, h)`.
    pub fn to_pixels(&self, width: u32, height: u32) -> (f32, f32, f32, f32) {
        let (fw, fh) = (width as f32, height as f32);
        (self.x * fw, self.y * fh, self.w * fw, self.h * fh)
    }
}

/// Playback options for `file:` sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlaybackConfig {
    /// Pace packets at the file's frame rate (like a camera). `false` reads
    /// as fast as the pipeline accepts, for offline evaluation.
    pub realtime: bool,
    /// Restart at the end of the file.
    #[serde(rename = "loop")]
    pub looped: bool,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self { realtime: true, looped: true }
    }
}

/// Capture options for `device:` sources (webcams, capture cards).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureConfig {
    /// libavdevice backend: `dshow` (Windows), `v4l2` (Linux),
    /// `avfoundation` (macOS). Defaults to the platform's.
    pub backend: Option<String>,
    /// Requested width.
    pub width: Option<u32>,
    /// Requested height.
    pub height: Option<u32>,
    /// Requested frame rate.
    pub fps: Option<f64>,
    /// Requested format: `mjpeg`, `yuyv422`, `nv12`, ... (device dependent).
    pub format: Option<String>,
}

/// Where a stream comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    /// RTSP camera or server.
    Rtsp,
    /// Local video file (MP4, MKV, MOV, raw H.264/H.265, MJPEG...).
    File(std::path::PathBuf),
    /// Capture device (webcam), e.g. `USB Camera` or `/dev/video0`.
    Device(String),
}

/// A camera (RTSP) or video-file source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StreamConfig {
    /// Unique identifier.
    pub id: String,
    /// `rtsp://` URL (credentials may be embedded or given separately), or
    /// `file:` followed by a path (`file:videos/a.mp4`, `file:///D:/a.mkv`).
    pub url: String,
    /// Optional user name (overrides the URL).
    pub username: Option<String>,
    /// Optional password (overrides the URL).
    pub password: Option<String>,
    /// RTSP transport.
    pub transport: Transport,
    /// Outputs fed by this stream.
    pub outputs: Vec<String>,
    /// Gating parameters.
    pub gating: GatingConfig,
    /// Decoder override.
    pub decoder: Option<DecoderPref>,
    /// Reconnect back-off upper bound in milliseconds.
    pub max_reconnect_ms: u64,
    /// Optional region of interest (normalised coordinates).
    pub region: Option<Region>,
    /// Playback options for `file:` sources.
    pub playback: PlaybackConfig,
    /// Capture options for `device:` sources.
    pub capture: CaptureConfig,
    /// Pin this stream to one of the engine's devices (multi-GPU); default:
    /// automatic placement.
    pub device: Option<Device>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            url: String::new(),
            username: None,
            password: None,
            transport: Transport::Tcp,
            outputs: Vec::new(),
            gating: GatingConfig::default(),
            decoder: None,
            max_reconnect_ms: 10_000,
            region: None,
            playback: PlaybackConfig::default(),
            capture: CaptureConfig::default(),
            device: None,
        }
    }
}

impl StreamConfig {
    /// Convenience constructor.
    pub fn new(id: impl Into<String>, url: impl Into<String>, outputs: &[&str]) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    /// Kind of source described by `url`.
    pub fn source_kind(&self) -> Result<SourceKind> {
        let lower = self.url.to_ascii_lowercase();
        if lower.starts_with("rtsp://") || lower.starts_with("rtsps://") {
            return Ok(SourceKind::Rtsp);
        }
        if lower.starts_with("file:") {
            let rest = &self.url[5..];
            // file:///C:/x, file:///home/x, file://C:/x, file:relative/x
            let path = match rest.strip_prefix("//") {
                Some(r) => {
                    let r = r.strip_prefix("localhost").unwrap_or(r);
                    let windows_drive =
                        r.len() > 2 && r.as_bytes()[0] == b'/' && r.as_bytes()[2] == b':';
                    if windows_drive { &r[1..] } else { r }
                }
                None => rest,
            };
            if path.is_empty() {
                return Err(Error::config(format!("stream '{}': empty file path", self.id)));
            }
            return Ok(SourceKind::File(path.replace("%20", " ").into()));
        }
        if lower.starts_with("device:") {
            let name = self.url[7..].trim();
            if name.is_empty() {
                return Err(Error::config(format!("stream '{}': empty device name", self.id)));
            }
            return Ok(SourceKind::Device(name.to_string()));
        }
        Err(Error::config(format!(
            "stream '{}': url must start with rtsp://, rtsps://, file: or device:",
            self.id
        )))
    }

    /// Validates the stream definition.
    pub fn validate(&self) -> Result<()> {
        validate_name(&self.id, "stream")?;
        self.source_kind()?;
        if let Some(r) = &self.region {
            r.validate().map_err(|e| Error::config(format!("stream '{}': {e}", self.id)))?;
        }
        if self.outputs.is_empty() {
            return Err(Error::config(format!("stream '{}' has no outputs", self.id)));
        }
        if self.gating.threshold <= 1.0 {
            return Err(Error::config(format!(
                "stream '{}': gating.threshold must be > 1",
                self.id
            )));
        }
        Ok(())
    }
}

/// MediaMTX auto-discovery: every ready path matching `paths` becomes a
/// stream feeding `outputs`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediaMtxConfig {
    /// MediaMTX control API base URL.
    pub api_url: String,
    /// RTSP base URL used to read the discovered paths.
    pub rtsp_url: String,
    /// Glob patterns (`*`, `?`) of path names to consume.
    pub paths: Vec<String>,
    /// Outputs fed by discovered streams.
    pub outputs: Vec<String>,
    /// Gating applied to discovered streams.
    pub gating: GatingConfig,
    /// Poll interval of the MediaMTX API.
    pub poll_interval_ms: u64,
    /// Credentials for the API and RTSP reads, if MediaMTX requires them.
    pub username: Option<String>,
    /// Password, see `username`.
    pub password: Option<String>,
}

impl Default for MediaMtxConfig {
    fn default() -> Self {
        Self {
            api_url: "http://127.0.0.1:9997".into(),
            rtsp_url: "rtsp://127.0.0.1:8554".into(),
            paths: vec!["*".into()],
            outputs: Vec::new(),
            gating: GatingConfig::default(),
            poll_interval_ms: 2000,
            username: None,
            password: None,
        }
    }
}

/// Minimal glob matcher supporting `*` and `?`.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0usize, 0usize, None::<usize>, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    const SAMPLE: &str = r#"
device: cuda:0
outputs:
  detector:
    tensor: { width: 640, height: 640, dtype: f16, resize: letterbox, normalize: unit }
    batch: { max_size: 8, max_wait_ms: 10 }
    keep_frame: true
  crops:
    tensor: { width: 112, height: 112, normalize: symmetric, resize: stretch }
streams:
  - id: cam1
    url: rtsp://127.0.0.1:8554/cam1
    outputs: [detector]
    gating: { mode: auto }
mediamtx:
  api_url: http://127.0.0.1:9997
  paths: ["cam*"]
  outputs: [detector]
"#;

    #[test]
    fn parse_sample() {
        let c = EngineConfig::from_yaml(SAMPLE).unwrap();
        assert_eq!(c.device, Device::Cuda(0));
        assert_eq!(c.outputs["detector"].tensor.dtype, DType::F16);
        assert_eq!(c.outputs["detector"].batch.max_size, 8);
        assert_eq!(c.streams[0].gating.mode, GatingMode::Auto);
        assert_eq!(c.mediamtx.as_ref().unwrap().paths, vec!["cam*"]);
        // round trip
        let again = EngineConfig::from_yaml(&c.to_yaml()).unwrap();
        assert_eq!(again, c);
    }

    #[test]
    fn rejects_unknown_output_and_fields() {
        let bad = "outputs: {a: {}}\nstreams: [{id: c, url: 'rtsp://x', outputs: [b]}]";
        assert!(EngineConfig::from_yaml(bad).is_err());
        assert!(EngineConfig::from_yaml("outputs: {a: {tensr: {}}}").is_err());
        let dup = "outputs: {a: {}}\nstreams: [{id: c, url: 'rtsp://x', outputs: [a]}, {id: c, url: 'rtsp://y', outputs: [a]}]";
        assert!(EngineConfig::from_yaml(dup).is_err());
        assert!(
            EngineConfig::from_yaml("streams: [{id: c, url: 'http://x', outputs: []}]").is_err()
        );
    }

    #[test]
    fn regions_and_file_urls() {
        let c = EngineConfig::from_yaml(
            "outputs: {a: {}}
streams:
  - {id: door, url: 'rtsp://x/1', outputs: [a], region: {x: 0.5, y: 0, w: 0.5, h: 1}}
  - {id: clip, url: 'file:videos/a.mp4', outputs: [a], playback: {realtime: false, loop: false}}
",
        )
        .unwrap();
        assert_eq!(c.streams[0].region.unwrap().to_pixels(1920, 1080), (960.0, 0.0, 960.0, 1080.0));
        assert_eq!(c.streams[1].source_kind().unwrap(), SourceKind::File("videos/a.mp4".into()));
        assert!(!c.streams[1].playback.looped && !c.streams[1].playback.realtime);
        let f = |u: &str| StreamConfig::new("s", u, &["a"]).source_kind().unwrap();
        assert_eq!(f("file:///D:/v/a%20b.mkv"), SourceKind::File("D:/v/a b.mkv".into()));
        assert_eq!(f("file:///home/u/a.mp4"), SourceKind::File("/home/u/a.mp4".into()));
        assert_eq!(f("rtsp://cam/1"), SourceKind::Rtsp);
        assert_eq!(f("device:USB Camera"), SourceKind::Device("USB Camera".into()));
        assert_eq!(f("device:/dev/video0"), SourceKind::Device("/dev/video0".into()));
        let bad = "outputs: {a: {}}
streams: [{id: s, url: 'rtsp://x', outputs: [a], region: {x: 0.8, y: 0, w: 0.5, h: 1}}]";
        assert!(EngineConfig::from_yaml(bad).is_err());
        assert!(
            EngineConfig::from_yaml(
                "outputs: {a: {}}
streams: [{id: s, url: 'http://x', outputs: [a]}]"
            )
            .is_err()
        );
    }

    #[test]
    fn globs() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("cam*", "cam12"));
        assert!(glob_match("site?/cam*", "site1/cam3"));
        assert!(!glob_match("cam*", "door1"));
        assert!(!glob_match("cam?", "cam12"));
    }
}
