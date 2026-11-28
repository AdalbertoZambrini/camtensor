//! Standalone camtensor service.
//!
//! * **Control plane** (HTTP/JSON, default `127.0.0.1:7070`): manage streams
//!   and outputs, read statistics and Prometheus metrics.
//! * **Data plane** (TCP, default `127.0.0.1:7071`): clients subscribe to an
//!   output and receive batches through a shared-memory ring (see
//!   `camtensor-client`).
//!
//! | Method | Path | Description |
//! |---|---|---|
//! | GET | `/health` | Liveness probe |
//! | GET | `/v1/info` | Version, device, capabilities |
//! | GET | `/v1/stats` | Engine statistics (JSON) |
//! | GET | `/metrics` | Prometheus metrics |
//! | GET | `/v1/config` | Current configuration |
//! | GET | `/v1/streams` | Stream statistics |
//! | POST | `/v1/streams` | Add a stream (`StreamConfig` JSON) |
//! | GET | `/v1/streams/{id}` | One stream |
//! | DELETE | `/v1/streams/{id}` | Remove a stream |
//! | POST | `/v1/streams/{id}/force_active` | `{"seconds": 10}` bypasses gating |
//! | GET | `/v1/outputs` | Output configurations |
//! | POST | `/v1/outputs/{name}` | Add an output (`OutputConfig` JSON) |

#![warn(missing_docs)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use camtensor::{Engine, Error, OutputConfig, StreamConfig};
use camtensor_client::protocol::{ClientMsg, RemoteItem, ServerMsg};

/// Service options.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Control plane address.
    pub http: SocketAddr,
    /// Data plane address.
    pub data: SocketAddr,
    /// Directory for shared-memory files (default: `/dev/shm` on Linux,
    /// the temp dir elsewhere).
    pub shm_dir: PathBuf,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            http: "127.0.0.1:7070".parse().expect("addr"),
            data: "127.0.0.1:7071".parse().expect("addr"),
            shm_dir: default_shm_dir(),
        }
    }
}

/// Default shared-memory directory.
pub fn default_shm_dir() -> PathBuf {
    let shm = PathBuf::from("/dev/shm");
    if cfg!(target_os = "linux") && shm.is_dir() { shm } else { std::env::temp_dir() }
}

/// Runs the control and data planes until `shutdown` resolves.
pub async fn serve(
    engine: Engine,
    opts: ServeOptions,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let http = TcpListener::bind(opts.http).await?;
    let data = TcpListener::bind(opts.data).await?;
    tracing::info!("control plane on http://{}", http.local_addr()?);
    tracing::info!("data plane on tcp://{}", data.local_addr()?);
    let data_task = tokio::spawn(data_plane(engine.clone(), data, opts.shm_dir.clone()));
    let app = router(engine);
    let res = axum::serve(http, app).with_graceful_shutdown(shutdown).await;
    data_task.abort();
    res
}

/// Builds the HTTP router (exposed for embedding and tests).
pub fn router(engine: Engine) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/info", get(info))
        .route("/v1/stats", get(stats))
        .route("/metrics", get(metrics))
        .route("/v1/config", get(config))
        .route("/v1/streams", get(list_streams).post(add_stream))
        .route("/v1/streams/{id}", get(get_stream).delete(delete_stream))
        .route("/v1/streams/{id}/force_active", post(force_active))
        .route("/v1/outputs", get(list_outputs))
        .route("/v1/outputs/{name}", post(add_output))
        .with_state(engine)
}

struct ApiError(Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = match &self.0 {
            Error::Config(_) => StatusCode::BAD_REQUEST,
            Error::NotFound(_) => StatusCode::NOT_FOUND,
            Error::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (code, Json(serde_json::json!({ "error": self.0.to_string() }))).into_response()
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        Self(e)
    }
}

type ApiResult<T> = Result<T, ApiError>;

async fn info(State(e): State<Engine>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "camtensor",
        "version": camtensor::VERSION,
        "device": e.device().to_string(),
        "devices": e.devices().iter().map(ToString::to_string).collect::<Vec<_>>(),
        "capabilities": camtensor::capabilities(),
    }))
}

async fn stats(State(e): State<Engine>) -> Json<camtensor::EngineStats> {
    Json(e.stats())
}

async fn metrics(State(e): State<Engine>) -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], e.stats().to_prometheus())
}

async fn config(State(e): State<Engine>) -> Json<camtensor::EngineConfig> {
    Json(e.config())
}

async fn list_streams(State(e): State<Engine>) -> Json<Vec<camtensor::StreamStats>> {
    Json(e.stats().streams)
}

async fn get_stream(
    State(e): State<Engine>,
    Path(id): Path<String>,
) -> ApiResult<Json<camtensor::StreamStats>> {
    e.stats()
        .streams
        .into_iter()
        .find(|s| s.id == id)
        .map(Json)
        .ok_or_else(|| Error::NotFound(format!("stream '{id}'")).into())
}

async fn add_stream(
    State(e): State<Engine>,
    Json(cfg): Json<StreamConfig>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let id = cfg.id.clone();
    tokio::task::spawn_blocking(move || e.add_stream(cfg))
        .await
        .map_err(|j| Error::Ingest(j.to_string()))??;
    Ok((StatusCode::CREATED, Json(serde_json::json!({ "id": id }))))
}

async fn delete_stream(State(e): State<Engine>, Path(id): Path<String>) -> ApiResult<StatusCode> {
    tokio::task::spawn_blocking(move || e.remove_stream(&id))
        .await
        .map_err(|j| Error::Ingest(j.to_string()))??;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct ForceActive {
    seconds: f64,
}

async fn force_active(
    State(e): State<Engine>,
    Path(id): Path<String>,
    Json(body): Json<ForceActive>,
) -> ApiResult<StatusCode> {
    e.force_active(&id, Duration::from_secs_f64(body.seconds.max(0.0)))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_outputs(State(e): State<Engine>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(e.config().outputs).unwrap_or_default())
}

async fn add_output(
    State(e): State<Engine>,
    Path(name): Path<String>,
    Json(cfg): Json<OutputConfig>,
) -> ApiResult<StatusCode> {
    e.add_output(&name, cfg)?;
    Ok(StatusCode::CREATED)
}

static SUB_ID: AtomicU64 = AtomicU64::new(0);

async fn data_plane(engine: Engine, listener: TcpListener, shm_dir: PathBuf) {
    loop {
        match listener.accept().await {
            Ok((sock, peer)) => {
                let engine = engine.clone();
                let dir = shm_dir.clone();
                tokio::spawn(async move {
                    if let Err(e) = subscriber(engine, sock, dir).await {
                        tracing::debug!(%peer, "subscriber ended: {e}");
                    }
                });
            }
            Err(e) => {
                tracing::warn!("data plane accept failed: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

async fn write_msg(w: &mut tokio::net::tcp::OwnedWriteHalf, m: &ServerMsg) -> std::io::Result<()> {
    let mut s = serde_json::to_string(m).map_err(std::io::Error::other)?;
    s.push('\n');
    w.write_all(s.as_bytes()).await
}

/// Removes the shared-memory file when the subscription ends.
struct ShmFile(PathBuf);

impl Drop for ShmFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn subscriber(engine: Engine, sock: TcpStream, shm_dir: PathBuf) -> std::io::Result<()> {
    sock.set_nodelay(true)?;
    let (rd, mut wr) = sock.into_split();
    let mut lines = BufReader::new(rd).lines();
    let first = lines.next_line().await?.unwrap_or_default();
    let (output, slots, device) = match serde_json::from_str::<ClientMsg>(&first) {
        Ok(ClientMsg::Subscribe { output, slots, device }) => (output, slots.clamp(1, 64), device),
        _ => {
            let m = ServerMsg::Error { message: "expected subscribe".into() };
            return write_msg(&mut wr, &m).await;
        }
    };
    let cfg = match engine.output_config(&output) {
        Ok(c) => c,
        Err(e) => return write_msg(&mut wr, &ServerMsg::Error { message: e.to_string() }).await,
    };
    let device: Option<camtensor::Device> = match device.map(|d| d.parse()).transpose() {
        Ok(d) => d,
        Err(e) => {
            let m = ServerMsg::Error { message: format!("{e}") };
            return write_msg(&mut wr, &m).await;
        }
    };
    let slot_bytes = cfg.tensor.image_bytes() * cfg.batch.max_size;
    let path = shm_dir.join(format!(
        "camtensor-{}-{}-{}.shm",
        std::process::id(),
        SUB_ID.fetch_add(1, Ordering::Relaxed),
        output.replace(['/', '\\'], "_")
    ));
    let guard = ShmFile(path.clone());
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;
    file.set_len((slot_bytes * slots) as u64)?;
    // SAFETY: the file was just created by us with the right size; the client
    // only reads slots after we announce them.
    let map = unsafe { memmap2::MmapMut::map_mut(&file)? };
    let map = Arc::new(parking_lot_free::Shared::new(map));

    write_msg(
        &mut wr,
        &ServerMsg::Subscribed {
            output: output.clone(),
            shm_path: path.to_string_lossy().into_owned(),
            slots,
            slot_bytes,
            tensor: cfg.tensor,
            max_batch: cfg.batch.max_size,
        },
    )
    .await?;

    // Free-slot bookkeeping: the reader task returns released slots.
    let (free_tx, mut free_rx) = tokio::sync::mpsc::unbounded_channel::<usize>();
    for i in 0..slots {
        let _ = free_tx.send(i);
    }
    let reader = tokio::spawn(async move {
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(ClientMsg::Release { slot }) = serde_json::from_str::<ClientMsg>(&line)
                && slot < slots
                && free_tx.send(slot).is_err()
            {
                break;
            }
        }
    });

    let result = async {
        let mut retry: Option<usize> = None;
        loop {
            let slot = match retry.take() {
                Some(s) => s,
                None => match free_rx.recv().await {
                    Some(s) => s,
                    None => return Ok(()),
                },
            };
            // Wait for a batch on a blocking thread (the engine API is synchronous).
            let (engine2, out2, map2) = (engine.clone(), output.clone(), map.clone());
            let msg = tokio::task::spawn_blocking(move || -> Result<Option<ServerMsg>, Error> {
                let wait = Some(Duration::from_millis(500));
                let got = match device {
                    Some(d) => engine2.recv_on(&out2, d, wait),
                    None => engine2.recv(&out2, wait),
                };
                let batch = match got {
                    Ok(b) => b,
                    Err(Error::Timeout) => return Ok(None),
                    Err(e) => return Err(e),
                };
                let t = batch.tensor();
                let bytes = t.byte_len();
                let off = slot * slot_bytes;
                map2.with_mut(|m| engine2.copy_to_host(t, &mut m[off..off + bytes]))?;
                Ok(Some(ServerMsg::Batch {
                    slot,
                    seq: batch.seq(),
                    shape: t.shape(),
                    dtype: format!("{:?}", t.dtype()).to_lowercase(),
                    bytes,
                    items: batch
                        .items()
                        .iter()
                        .map(|i| RemoteItem { valid: i.valid, meta: i.meta.clone() })
                        .collect(),
                }))
            })
            .await
            .map_err(std::io::Error::other)?;
            match msg {
                Ok(Some(m)) => write_msg(&mut wr, &m).await?,
                Ok(None) => {
                    // No batch yet: give the slot back and try again.
                    if reader.is_finished() {
                        return Ok(());
                    }
                    retry = Some(slot);
                }
                Err(Error::Closed) => return Ok(()),
                Err(e) => {
                    write_msg(&mut wr, &ServerMsg::Error { message: e.to_string() }).await?;
                    return Ok(());
                }
            }
        }
    }
    .await;
    reader.abort();
    drop(guard);
    result
}

mod parking_lot_free {
    //! A tiny mutex-protected shared mapping.
    use std::sync::Mutex;

    pub(crate) struct Shared<T>(Mutex<T>);

    impl<T> Shared<T> {
        pub(crate) fn new(v: T) -> Self {
            Self(Mutex::new(v))
        }
        pub(crate) fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
            let mut g = self.0.lock().unwrap_or_else(|p| p.into_inner());
            f(&mut g)
        }
    }
}
