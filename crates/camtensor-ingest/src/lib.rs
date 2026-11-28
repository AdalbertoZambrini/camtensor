//! RTSP ingest (retina) producing access units for camtensor.
//!
//! [`RtspSource::spawn`] starts a task that connects to an RTSP server (an
//! IP camera or MediaMTX), sets up the first H.264, H.265 or JPEG video
//! stream and sends [`SourceEvent`]s on a bounded channel:
//!
//! * [`SourceEvent::Connected`] after every successful `PLAY`;
//! * [`SourceEvent::ParamsChanged`] when the parameter sets change;
//! * [`SourceEvent::Packet`] for every access unit, Annex B for H.26x with
//!   the parameter sets prepended on keyframes, a full JPEG for MJPEG;
//! * [`SourceEvent::Disconnected`] for every failure, followed by a
//!   reconnect with exponential back-off (500 ms up to `max_reconnect_ms`).
//!
//! Packets are delivered with `try_send`: when the channel is full the packet
//! and all following non-key packets are dropped until the next keyframe,
//! which is delivered with `loss = true`. Control events are never dropped.
//!
//! `pts` is in 90 kHz ticks and stays monotonic across reconnects (a new
//! session continues from the previous one, advanced by the outage).
//! `capture_time_ns` is derived from RTCP sender reports once one arrived.

#![forbid(unsafe_code)]

// Version pin only (see Cargo.toml).
use pretty_hex as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use camtensor_core::{EncodedPacket, StreamConfig, StreamParams, Transport};

mod convert;
mod source;

/// Connection settings of one RTSP source.
#[derive(Clone, PartialEq, Eq)]
pub struct SourceConfig {
    /// Stream identifier copied into every packet.
    pub stream_id: Arc<str>,
    /// `rtsp://` URL; credentials may be embedded (percent-encoded).
    pub url: String,
    /// User name (overrides the URL).
    pub username: Option<String>,
    /// Password (overrides the URL).
    pub password: Option<String>,
    /// RTP transport.
    pub transport: Transport,
    /// Upper bound of the reconnect back-off.
    pub max_reconnect_ms: u64,
    /// A session that delivers no video for this long is treated as failed.
    pub data_timeout: Duration,
}

impl SourceConfig {
    /// Defaults: TCP, 10 s back-off cap, 10 s data timeout.
    pub fn new(stream_id: &str, url: impl Into<String>) -> Self {
        Self {
            stream_id: Arc::from(stream_id),
            url: url.into(),
            username: None,
            password: None,
            transport: Transport::Tcp,
            max_reconnect_ms: 10_000,
            data_timeout: Duration::from_secs(10),
        }
    }
}

impl std::fmt::Debug for SourceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceConfig")
            .field("stream_id", &self.stream_id)
            .field("url", &redact_url(&self.url))
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "<redacted>"))
            .field("transport", &self.transport)
            .field("max_reconnect_ms", &self.max_reconnect_ms)
            .field("data_timeout", &self.data_timeout)
            .finish()
    }
}

fn redact_url(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut u) if u.password().is_some() => {
            let _ = u.set_password(Some("redacted"));
            u.to_string()
        }
        _ => url.to_owned(),
    }
}

impl From<&StreamConfig> for SourceConfig {
    fn from(s: &StreamConfig) -> Self {
        Self {
            stream_id: Arc::from(s.id.as_str()),
            url: s.url.clone(),
            username: s.username.clone(),
            password: s.password.clone(),
            transport: s.transport,
            max_reconnect_ms: s.max_reconnect_ms,
            data_timeout: Duration::from_secs(10),
        }
    }
}

/// A decoded frame produced directly by a source.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// The picture (host memory).
    pub frame: camtensor_core::Frame,
    /// Monotonic per-stream sequence number.
    pub seq: u64,
    /// Wall-clock time (ns since Unix epoch) when it was captured/received.
    pub recv_time_ns: u64,
}

/// Events produced by a source.
#[derive(Debug, Clone)]
pub enum SourceEvent {
    /// A session started (also after every reconnect). Parameters may be
    /// incomplete (zero size, empty extradata) when the server only sends
    /// them in-band; a `ParamsChanged` follows once they are known.
    Connected(StreamParams),
    /// The stream parameters changed mid-session.
    ParamsChanged(StreamParams),
    /// One access unit.
    Packet(EncodedPacket),
    /// An already decoded frame (capture devices delivering raw video).
    Frame(RawFrame),
    /// The session failed; a reconnect follows.
    Disconnected {
        /// Human-readable cause.
        reason: String,
    },
}

/// Snapshot of a source's counters.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceStats {
    /// `true` while a session is playing.
    pub connected: bool,
    /// Packets received from the server (delivered or dropped).
    pub packets: u64,
    /// Payload bytes received.
    pub bytes: u64,
    /// Packets dropped because the channel was full.
    pub dropped: u64,
    /// Reconnect attempts.
    pub reconnects: u64,
    /// Cause of the last failure.
    pub last_error: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct Shared {
    pub connected: AtomicBool,
    pub packets: AtomicU64,
    pub bytes: AtomicU64,
    pub dropped: AtomicU64,
    pub reconnects: AtomicU64,
    pub last_error: Mutex<Option<String>>,
}

/// RTSP source factory.
#[derive(Debug)]
pub struct RtspSource;

impl RtspSource {
    /// Spawns the source task on `handle`. Events go to `tx`; the task ends
    /// when the handle is stopped or dropped, or when the receiver is
    /// dropped.
    pub fn spawn(
        handle: &tokio::runtime::Handle,
        cfg: SourceConfig,
        tx: mpsc::Sender<SourceEvent>,
    ) -> SourceHandle {
        let shared = Arc::new(Shared::default());
        let (stop_tx, stop_rx) = watch::channel(false);
        let stream_id = cfg.stream_id.clone();
        let task = handle.spawn(source::run(cfg, tx, shared.clone(), stop_rx));
        SourceHandle { stream_id, stop: stop_tx, shared, task: Some(task) }
    }
}

/// Handle of a running source. Dropping it stops the source.
#[derive(Debug)]
pub struct SourceHandle {
    stream_id: Arc<str>,
    stop: watch::Sender<bool>,
    shared: Arc<Shared>,
    task: Option<JoinHandle<()>>,
}

impl SourceHandle {
    /// Stream identifier.
    pub fn stream_id(&self) -> &Arc<str> {
        &self.stream_id
    }

    /// Asks the source to stop (tears the RTSP session down).
    pub fn stop(&self) {
        self.stop.send_replace(true);
    }

    /// `true` once the task has ended.
    pub fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(|t| t.is_finished())
    }

    /// Stops the source and waits for its task to end.
    pub async fn join(mut self) {
        self.stop();
        if let Some(t) = self.task.take() {
            let _ = t.await;
        }
    }

    /// Counter snapshot.
    pub fn stats(&self) -> SourceStats {
        let s = &self.shared;
        SourceStats {
            connected: s.connected.load(Ordering::Relaxed),
            packets: s.packets.load(Ordering::Relaxed),
            bytes: s.bytes.load(Ordering::Relaxed),
            dropped: s.dropped.load(Ordering::Relaxed),
            reconnects: s.reconnects.load(Ordering::Relaxed),
            last_error: s.last_error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }
}

impl Drop for SourceHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_stream_config() {
        let mut s = StreamConfig::new("cam1", "rtsp://u:secret@h/p", &["o"]);
        s.transport = Transport::Udp;
        s.max_reconnect_ms = 3000;
        let c = SourceConfig::from(&s);
        assert_eq!(&*c.stream_id, "cam1");
        assert_eq!(c.transport, Transport::Udp);
        assert_eq!(c.max_reconnect_ms, 3000);
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("secret"), "{dbg}");
    }

    #[tokio::test]
    async fn stop_on_drop_and_unreachable_reports_errors() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut cfg = SourceConfig::new("x", "rtsp://127.0.0.1:1/none");
        cfg.max_reconnect_ms = 500;
        let h = RtspSource::spawn(&tokio::runtime::Handle::current(), cfg, tx);
        match tokio::time::timeout(Duration::from_secs(15), rx.recv()).await {
            Ok(Some(SourceEvent::Disconnected { reason })) => {
                assert!(reason.contains("DESCRIBE"), "{reason}")
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(h.stats().last_error.is_some());
        assert!(!h.stats().connected);
        drop(h);
        // The task ends and drops its sender.
        let end = tokio::time::timeout(Duration::from_secs(5), async {
            while rx.recv().await.is_some() {}
        })
        .await;
        assert!(end.is_ok());
    }
}
