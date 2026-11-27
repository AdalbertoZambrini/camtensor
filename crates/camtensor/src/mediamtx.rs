//! MediaMTX auto-discovery: polls the MediaMTX control API and turns every
//! ready path matching the configured globs into a stream.

use std::collections::HashSet;
use std::sync::{Arc, Weak};
use std::time::Duration;

use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use camtensor_core::config::glob_match;
use camtensor_core::{Error, MediaMtxConfig, Result, StreamConfig};

use crate::engine::{Engine, Inner};

/// Running discovery task.
pub(crate) struct Discovery {
    task: tokio::task::JoinHandle<()>,
}

impl Discovery {
    pub(crate) fn start(engine: Engine, cfg: MediaMtxConfig) -> Result<Self> {
        if cfg.outputs.is_empty() {
            return Err(Error::config("mediamtx: no outputs configured"));
        }
        let weak = Arc::downgrade(&engine.inner);
        let task = engine.runtime().spawn(run(weak, cfg));
        Ok(Self { task })
    }

    pub(crate) fn stop(self) {
        self.task.abort();
    }
}

#[derive(Debug, Deserialize)]
struct PathList {
    #[serde(default)]
    items: Vec<PathItem>,
}

#[derive(Debug, Deserialize)]
struct PathItem {
    name: String,
    #[serde(default)]
    ready: bool,
}

async fn run(engine: Weak<Inner>, cfg: MediaMtxConfig) {
    let mut ours: HashSet<String> = HashSet::new();
    let interval = Duration::from_millis(cfg.poll_interval_ms.max(200));
    let url = format!("{}/v3/paths/list?itemsPerPage=10000", cfg.api_url.trim_end_matches('/'));
    loop {
        match http_get(&url, cfg.username.as_deref(), cfg.password.as_deref()).await {
            Ok(body) => match serde_json::from_str::<PathList>(&body) {
                Ok(list) => {
                    let Some(inner) = engine.upgrade() else { return };
                    let engine = Engine { inner };
                    reconcile(&engine, &cfg, &list, &mut ours);
                }
                Err(e) => tracing::warn!("mediamtx: bad API response: {e}"),
            },
            Err(e) => tracing::debug!("mediamtx: API unreachable: {e}"),
        }
        if engine.strong_count() == 0 {
            return;
        }
        tokio::time::sleep(interval).await;
    }
}

fn reconcile(engine: &Engine, cfg: &MediaMtxConfig, list: &PathList, ours: &mut HashSet<String>) {
    let ready: HashSet<&str> = list
        .items
        .iter()
        .filter(|p| p.ready && cfg.paths.iter().any(|g| glob_match(g, &p.name)))
        .map(|p| p.name.as_str())
        .collect();
    // Remove streams whose path went away.
    let gone: Vec<String> = ours.iter().filter(|n| !ready.contains(n.as_str())).cloned().collect();
    for name in gone {
        tracing::info!(path = %name, "mediamtx: path no longer ready, removing stream");
        let _ = engine.remove_stream(&name);
        ours.remove(&name);
    }
    let existing: HashSet<String> = engine.streams().into_iter().collect();
    for name in ready {
        if ours.contains(name) || existing.contains(name) {
            continue;
        }
        let sc = StreamConfig {
            id: name.to_string(),
            url: format!("{}/{}", cfg.rtsp_url.trim_end_matches('/'), name),
            username: cfg.username.clone(),
            password: cfg.password.clone(),
            outputs: cfg.outputs.clone(),
            gating: cfg.gating,
            ..Default::default()
        };
        match engine.add_stream(sc) {
            Ok(()) => {
                tracing::info!(path = %name, "mediamtx: stream discovered");
                ours.insert(name.to_string());
            }
            Err(e) => tracing::warn!(path = %name, "mediamtx: cannot add stream: {e}"),
        }
    }
}

/// Minimal HTTP/1.1 GET over plain TCP (the MediaMTX API is local).
pub(crate) async fn http_get(url: &str, user: Option<&str>, pass: Option<&str>) -> Result<String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| Error::config("mediamtx api_url must start with http://"))?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let addr = if hostport.contains(':') { hostport.to_string() } else { format!("{hostport}:80") };
    let fut = async {
        let mut sock = tokio::net::TcpStream::connect(&addr).await?;
        let mut req = format!(
            "GET {path} HTTP/1.1\r\nHost: {hostport}\r\nUser-Agent: camtensor\r\nAccept: application/json\r\nConnection: close\r\n"
        );
        if let Some(u) = user {
            let token = base64(format!("{u}:{}", pass.unwrap_or("")).as_bytes());
            req.push_str(&format!("Authorization: Basic {token}\r\n"));
        }
        req.push_str("\r\n");
        sock.write_all(req.as_bytes()).await?;
        let mut buf = Vec::new();
        sock.read_to_end(&mut buf).await?;
        Ok::<_, std::io::Error>(buf)
    };
    let buf = tokio::time::timeout(Duration::from_secs(5), fut)
        .await
        .map_err(|_| Error::Timeout)?
        .map_err(Error::Io)?;
    parse_response(&buf)
}

fn parse_response(buf: &[u8]) -> Result<String> {
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| Error::Ingest("malformed HTTP response".into()))?;
    let head = String::from_utf8_lossy(&buf[..split]).to_string();
    let body = &buf[split + 4..];
    let status = head.split_whitespace().nth(1).unwrap_or("");
    if status != "200" {
        return Err(Error::Ingest(format!("HTTP {status}")));
    }
    let chunked = head.lines().any(|l| {
        l.to_ascii_lowercase().starts_with("transfer-encoding:")
            && l.to_ascii_lowercase().contains("chunked")
    });
    let body = if chunked { dechunk(body)? } else { body.to_vec() };
    String::from_utf8(body).map_err(|_| Error::Ingest("non UTF-8 body".into()))
}

fn dechunk(mut b: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end = b
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| Error::Ingest("bad chunk".into()))?;
        let size_str = String::from_utf8_lossy(&b[..line_end]);
        let size = usize::from_str_radix(size_str.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| Error::Ingest("bad chunk size".into()))?;
        b = &b[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if b.len() < size + 2 {
            return Err(Error::Ingest("truncated chunk".into()));
        }
        out.extend_from_slice(&b[..size]);
        b = &b[size + 2..];
    }
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_vectors() {
        assert_eq!(base64(b"user:pass"), "dXNlcjpwYXNz");
        assert_eq!(base64(b"a"), "YQ==");
        assert_eq!(base64(b"ab"), "YWI=");
    }

    #[test]
    fn parses_plain_and_chunked() {
        let plain = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(parse_response(plain).unwrap(), "{}");
        let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n{\"a\"\r\n3\r\n:1}\r\n0\r\n\r\n";
        assert_eq!(parse_response(chunked).unwrap(), "{\"a\":1}");
        assert!(parse_response(b"HTTP/1.1 401 Unauthorized\r\n\r\n").is_err());
    }

    #[test]
    fn parses_path_list() {
        let body = r#"{"pageCount":1,"itemCount":2,"items":[{"name":"cam1","ready":true,"tracks":["H264"]},{"name":"cam2","ready":false}]}"#;
        let l: PathList = serde_json::from_str(body).unwrap();
        assert_eq!(l.items.len(), 2);
        assert!(l.items[0].ready && !l.items[1].ready);
    }
}
