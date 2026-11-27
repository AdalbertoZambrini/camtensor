//! The [`Engine`]: owns outputs, streams, the async runtime and the device.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Mutex, RwLock};

use camtensor_core::{
    Device, EngineConfig, Error, Frame, OutputConfig, Result, Roi, StreamConfig, TensorSpec,
};

use crate::device::Backend;
use crate::output::Output;
use crate::stats::EngineStats;
use crate::stream::{self, StreamHandle};
use crate::tensor::{Batch, Tensor};

pub(crate) struct Inner {
    pub(crate) cfg: Mutex<EngineConfig>,
    pub(crate) backend: Backend,
    rt: tokio::runtime::Runtime,
    outputs: RwLock<BTreeMap<String, Arc<Output>>>,
    streams: Mutex<HashMap<String, StreamHandle>>,
    discovery: Mutex<Option<crate::mediamtx::Discovery>>,
}

/// Camera streams in, tensors out.
///
/// ```no_run
/// use camtensor::{Engine, EngineConfig};
/// let engine = Engine::new(EngineConfig::from_yaml(r#"
/// outputs:
///   detector:
///     tensor: { width: 640, height: 640, dtype: f16 }
///     batch: { max_size: 4, max_wait_ms: 20 }
/// streams:
///   - { id: cam1, url: "rtsp://127.0.0.1:8554/cam1", outputs: [detector] }
/// "#)?)?;
/// for batch in engine.subscribe("detector")? {
///     println!("{} images, shape {:?}", batch.len(), batch.tensor().shape());
/// }
/// # Ok::<(), camtensor::Error>(())
/// ```
///
/// `Engine` is cheap to clone (shared handle) and thread-safe.
#[derive(Clone)]
pub struct Engine {
    pub(crate) inner: Arc<Inner>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").field("device", &self.device()).finish_non_exhaustive()
    }
}

impl Engine {
    /// Creates an engine, then starts every output, stream and the MediaMTX
    /// discovery declared in `cfg`.
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        cfg.validate()?;
        let backend = Backend::resolve(cfg.device)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("ct-io")
            .enable_all()
            .build()
            .map_err(Error::Io)?;
        tracing::info!(device = %backend.device(), "camtensor engine starting");
        let engine = Engine {
            inner: Arc::new(Inner {
                cfg: Mutex::new(EngineConfig { streams: Vec::new(), ..cfg.clone() }),
                backend,
                rt,
                outputs: RwLock::new(BTreeMap::new()),
                streams: Mutex::new(HashMap::new()),
                discovery: Mutex::new(None),
            }),
        };
        for (name, out) in &cfg.outputs {
            engine.add_output(name, out.clone())?;
        }
        for s in &cfg.streams {
            engine.add_stream(s.clone())?;
        }
        if let Some(m) = &cfg.mediamtx {
            let d = crate::mediamtx::Discovery::start(engine.clone(), m.clone())?;
            *engine.inner.discovery.lock() = Some(d);
        }
        Ok(engine)
    }

    /// Parses a YAML/JSON configuration file and creates the engine.
    pub fn from_config_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::new(EngineConfig::from_path(path)?)
    }

    /// Device used for decoding and, by default, for output tensors.
    pub fn device(&self) -> Device {
        self.inner.backend.device()
    }

    /// Adds a named output. Fails if the name exists.
    pub fn add_output(&self, name: &str, cfg: OutputConfig) -> Result<()> {
        let device = self.inner.backend.output_device(cfg.device)?;
        let mut outputs = self.inner.outputs.write();
        if outputs.contains_key(name) {
            return Err(Error::config(format!("output '{name}' already exists")));
        }
        let out = Output::new(
            name,
            cfg.clone(),
            device,
            self.inner.backend.alloc_fn(device),
            self.inner.backend.zero_fn(device),
        )?;
        outputs.insert(name.to_string(), out);
        self.inner.cfg.lock().outputs.insert(name.to_string(), cfg);
        Ok(())
    }

    /// Names of the configured outputs.
    pub fn outputs(&self) -> Vec<String> {
        self.inner.outputs.read().keys().cloned().collect()
    }

    fn output(&self, name: &str) -> Result<Arc<Output>> {
        self.inner
            .outputs
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("output '{name}'")))
    }

    /// Output configuration.
    pub fn output_config(&self, name: &str) -> Result<OutputConfig> {
        Ok(self.output(name)?.cfg.clone())
    }

    /// Device of an output's tensors.
    pub fn output_device(&self, name: &str) -> Result<Device> {
        Ok(self.output(name)?.device)
    }

    /// Starts consuming a stream. Fails if the id exists or references an
    /// unknown output.
    pub fn add_stream(&self, cfg: StreamConfig) -> Result<()> {
        cfg.validate()?;
        let outputs = cfg.outputs.iter().map(|o| self.output(o)).collect::<Result<Vec<_>>>()?;
        let mut streams = self.inner.streams.lock();
        if streams.contains_key(&cfg.id) {
            return Err(Error::config(format!("stream '{}' already exists", cfg.id)));
        }
        let default_decoder = self.inner.cfg.lock().decoder;
        let handle = stream::spawn(
            cfg.clone(),
            default_decoder,
            outputs,
            self.inner.backend.clone(),
            self.inner.rt.handle(),
        )?;
        streams.insert(cfg.id.clone(), handle);
        self.inner.cfg.lock().streams.push(cfg);
        Ok(())
    }

    /// Stops and removes a stream.
    pub fn remove_stream(&self, id: &str) -> Result<()> {
        let handle = self.inner.streams.lock().remove(id);
        match handle {
            Some(mut h) => {
                h.stop();
                self.inner.cfg.lock().streams.retain(|s| s.id != id);
                Ok(())
            }
            None => Err(Error::NotFound(format!("stream '{id}'"))),
        }
    }

    /// Ids of the running streams.
    pub fn streams(&self) -> Vec<String> {
        let mut v: Vec<String> = self.inner.streams.lock().keys().cloned().collect();
        v.sort();
        v
    }

    /// Configuration of a running stream.
    pub fn stream_config(&self, id: &str) -> Result<StreamConfig> {
        self.inner
            .streams
            .lock()
            .get(id)
            .map(|h| h.shared.cfg.clone())
            .ok_or_else(|| Error::NotFound(format!("stream '{id}'")))
    }

    /// Forces full decoding of a stream for `duration`, bypassing gating
    /// (e.g. while a tracked face is visible).
    pub fn force_active(&self, id: &str, duration: Duration) -> Result<()> {
        let streams = self.inner.streams.lock();
        let h = streams.get(id).ok_or_else(|| Error::NotFound(format!("stream '{id}'")))?;
        h.shared.force_active(duration);
        Ok(())
    }

    /// Waits for the next batch of `output` (`None` = wait forever).
    ///
    /// Each output is a single queue: concurrent receivers share (split)
    /// its batches.
    pub fn recv(&self, output: &str, timeout: Option<Duration>) -> Result<Batch> {
        self.output(output)?.recv(timeout)
    }

    /// Returns the next batch if one is ready.
    pub fn try_recv(&self, output: &str) -> Result<Option<Batch>> {
        Ok(self.output(output)?.try_recv())
    }

    /// Blocking iterator over the batches of `output`; ends when the engine
    /// shuts down.
    pub fn subscribe(&self, output: &str) -> Result<Subscription> {
        Ok(Subscription { out: self.output(output)? })
    }

    /// Extracts regions of `frame` (from a batch item kept with
    /// `keep_frame: true`) as a tensor `[rois.len(), ...]` formatted by
    /// `spec` — e.g. aligned 112x112 ArcFace crops from five landmarks.
    ///
    /// The tensor is produced on `device` (default: the engine device).
    pub fn crop(
        &self,
        frame: &Frame,
        rois: &[Roi],
        spec: &TensorSpec,
        device: Option<Device>,
    ) -> Result<Tensor> {
        spec.validate()?;
        if rois.is_empty() {
            return Err(Error::config("crop: no regions given"));
        }
        let device = self.inner.backend.output_device(device)?;
        let plans = rois
            .iter()
            .map(|r| camtensor_preprocess::plan_roi(r, spec).map(|p| p.sampling))
            .collect::<Result<Vec<_>>>()?;
        let mut ctx = self.inner.backend.worker()?;
        let src = ctx.frame_for(frame, device)?;
        ctx.crop(&self.inner.backend, &src, &plans, spec, device)
    }

    /// Snapshot of runtime statistics.
    pub fn stats(&self) -> EngineStats {
        let streams = {
            let s = self.inner.streams.lock();
            let mut v: Vec<_> = s.values().map(|h| h.stats()).collect();
            v.sort_by(|a, b| a.id.cmp(&b.id));
            v
        };
        let outputs =
            self.inner.outputs.read().iter().map(|(k, o)| (k.clone(), o.stats())).collect();
        EngineStats { device: self.device().to_string(), streams, outputs }
    }

    /// Current configuration (outputs, running streams, discovery).
    pub fn config(&self) -> EngineConfig {
        self.inner.cfg.lock().clone()
    }

    /// Stops discovery and every stream and wakes up all consumers.
    pub fn shutdown(&self) {
        if let Some(d) = self.inner.discovery.lock().take() {
            d.stop();
        }
        let streams: Vec<StreamHandle> = self.inner.streams.lock().drain().map(|(_, h)| h).collect();
        drop(streams);
        for o in self.inner.outputs.read().values() {
            o.close();
        }
    }

    pub(crate) fn runtime(&self) -> &tokio::runtime::Handle {
        self.inner.rt.handle()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.streams.get_mut().clear();
        for o in self.outputs.get_mut().values() {
            o.close();
        }
    }
}

/// Blocking iterator over an output's batches.
#[derive(Debug)]
pub struct Subscription {
    out: Arc<Output>,
}

impl Subscription {
    /// Waits for the next batch with a timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Batch> {
        self.out.recv(Some(timeout))
    }
}

impl Iterator for Subscription {
    type Item = Batch;
    fn next(&mut self) -> Option<Batch> {
        self.out.recv(None).ok()
    }
}
