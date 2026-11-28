//! The [`Engine`]: owns outputs, streams, the async runtime and the device.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Mutex, RwLock};

use camtensor_core::{
    Device, EngineConfig, Error, Frame, OutputConfig, Result, Roi, StreamConfig, TensorSpec,
};

use crate::device::Backend;
use crate::output::{Output, OutputGroup, Signal};
use crate::stats::EngineStats;
use crate::stream::{self, Loads, Placement, StreamHandle};
use crate::tensor::{Batch, Tensor};

pub(crate) struct Inner {
    pub(crate) cfg: Mutex<EngineConfig>,
    /// One backend per engine device (multi-GPU); never empty.
    pub(crate) backends: Vec<Backend>,
    loads: Arc<Loads>,
    rt: tokio::runtime::Runtime,
    outputs: RwLock<BTreeMap<String, Arc<OutputGroup>>>,
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
        let backends = resolve_backends(&cfg)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("ct-io")
            .enable_all()
            .build()
            .map_err(Error::Io)?;
        let names: Vec<String> = backends.iter().map(|b| b.device().to_string()).collect();
        tracing::info!(devices = ?names, "camtensor engine starting");
        let engine = Engine {
            inner: Arc::new(Inner {
                cfg: Mutex::new(EngineConfig { streams: Vec::new(), ..cfg.clone() }),
                loads: Arc::new(Loads::new(backends.len())),
                backends,
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

    /// Primary device (the first of [`Engine::devices`]).
    pub fn device(&self) -> Device {
        self.inner.backends[0].device()
    }

    /// Every device of the engine (several with multi-GPU).
    pub fn devices(&self) -> Vec<Device> {
        self.inner.backends.iter().map(Backend::device).collect()
    }

    fn backend_index(&self, device: Device) -> Option<usize> {
        self.inner.backends.iter().position(|b| b.device() == device)
    }

    /// Backend owning `device` memory, or the primary one.
    fn backend_for(&self, device: Device) -> &Backend {
        self.backend_index(device).map_or(&self.inner.backends[0], |i| &self.inner.backends[i])
    }

    /// Adds a named output (one shard per engine device). Fails if the name
    /// exists.
    pub fn add_output(&self, name: &str, cfg: OutputConfig) -> Result<()> {
        let mut outputs = self.inner.outputs.write();
        if outputs.contains_key(name) {
            return Err(Error::config(format!("output '{name}' already exists")));
        }
        let signal = Arc::new(Signal::default());
        let mut shards = Vec::with_capacity(self.inner.backends.len());
        for b in &self.inner.backends {
            let device = b.output_device(cfg.device)?;
            shards.push(Output::new(
                name,
                cfg.clone(),
                device,
                b.alloc_fn(device),
                b.zero_fn(device),
                Some(signal.clone()),
            )?);
        }
        outputs.insert(name.to_string(), Arc::new(OutputGroup::new(cfg.clone(), shards, signal)));
        self.inner.cfg.lock().outputs.insert(name.to_string(), cfg);
        Ok(())
    }

    /// Names of the configured outputs.
    pub fn outputs(&self) -> Vec<String> {
        self.inner.outputs.read().keys().cloned().collect()
    }

    fn output(&self, name: &str) -> Result<Arc<OutputGroup>> {
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

    /// Device of an output's tensors (of its first shard with multi-GPU).
    pub fn output_device(&self, name: &str) -> Result<Device> {
        Ok(self.output(name)?.shards[0].device)
    }

    /// Starts consuming a stream. Fails if the id exists or references an
    /// unknown output.
    pub fn add_stream(&self, cfg: StreamConfig) -> Result<()> {
        cfg.validate()?;
        let groups = cfg.outputs.iter().map(|o| self.output(o)).collect::<Result<Vec<_>>>()?;
        let mut streams = self.inner.streams.lock();
        if streams.contains_key(&cfg.id) {
            return Err(Error::config(format!("stream '{}' already exists", cfg.id)));
        }
        // Placement: pinned device, or the least loaded one (sticky).
        let index = match cfg.device {
            Some(d) => self.backend_index(d).ok_or_else(|| {
                Error::config(format!(
                    "stream '{}' is pinned to {d}, which is not one of the engine devices {:?}",
                    cfg.id,
                    self.devices().iter().map(ToString::to_string).collect::<Vec<_>>()
                ))
            })?,
            None => self.inner.loads.least_loaded(),
        };
        let outputs = groups.iter().map(|g| g.shards[index].clone()).collect();
        let default_decoder = self.inner.cfg.lock().decoder;
        let handle = stream::spawn(
            cfg.clone(),
            default_decoder,
            outputs,
            self.inner.backends[index].clone(),
            Placement { index, loads: self.inner.loads.clone() },
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
    /// (e.g. while an object of interest is in view).
    pub fn force_active(&self, id: &str, duration: Duration) -> Result<()> {
        let streams = self.inner.streams.lock();
        let h = streams.get(id).ok_or_else(|| Error::NotFound(format!("stream '{id}'")))?;
        h.shared.force_active(duration);
        Ok(())
    }

    /// Waits for the next batch of `output` (`None` = wait forever), from
    /// any device. The batch tensor's device tells which one.
    ///
    /// Each output is a single queue: concurrent receivers share (split)
    /// its batches.
    pub fn recv(&self, output: &str, timeout: Option<Duration>) -> Result<Batch> {
        self.output(output)?.recv(timeout)
    }

    /// Waits for the next batch of `output` produced on `device` (one
    /// consumer per GPU in multi-GPU setups).
    pub fn recv_on(
        &self,
        output: &str,
        device: Device,
        timeout: Option<Duration>,
    ) -> Result<Batch> {
        self.output(output)?.shard_for(device)?.recv(timeout)
    }

    /// Returns the next batch if one is ready.
    pub fn try_recv(&self, output: &str) -> Result<Option<Batch>> {
        Ok(self.output(output)?.try_recv())
    }

    /// Blocking iterator over the batches of `output`; ends when the engine
    /// shuts down.
    pub fn subscribe(&self, output: &str) -> Result<Subscription> {
        Ok(Subscription { group: self.output(output)?, device: None })
    }

    /// Like [`Engine::subscribe`], limited to batches produced on `device`.
    pub fn subscribe_on(&self, output: &str, device: Device) -> Result<Subscription> {
        let group = self.output(output)?;
        group.shard_for(device)?;
        Ok(Subscription { group, device: Some(device) })
    }

    /// Extracts regions of `frame` (from a batch item kept with
    /// `keep_frame: true`) as a tensor `[rois.len(), ...]` formatted by
    /// `spec`, e.g. 112x112 crops aligned from five keypoints for a second-stage model.
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
        // Work on the device that holds the frame (multi-GPU).
        let home = match &frame.data {
            camtensor_core::FrameData::Device(m) => Device::Cuda(m.device_ordinal()),
            // Host frames stay on the CPU when the engine has a CPU device;
            // otherwise the primary device (CPU-decoded frames on a GPU engine).
            camtensor_core::FrameData::Host(_) => device.unwrap_or(Device::Cpu),
        };
        let backend = self.backend_for(home);
        let device = backend.output_device(device)?;
        let plans = rois
            .iter()
            .map(|r| camtensor_preprocess::plan_roi(r, spec).map(|p| p.sampling))
            .collect::<Result<Vec<_>>>()?;
        let mut ctx = backend.worker()?;
        let src = ctx.frame_for(frame, device)?;
        ctx.crop(backend, &src, &plans, spec, device)
    }

    /// Encodes a kept frame (or a region of it) as a JPEG image.
    ///
    /// The frame is converted to RGB by the same fused kernel (on the GPU
    /// when the frame lives there), copied to the host once and encoded with
    /// a SIMD JPEG encoder. Typical cost for a 1080p frame: a few
    /// milliseconds; nothing is decoded again.
    pub fn snapshot(&self, frame: &Frame, opts: &SnapshotOptions) -> Result<Vec<u8>> {
        let (fw, fh) = (frame.desc.width as f32, frame.desc.height as f32);
        let (x, y, w, h) = match opts.region {
            Some([x, y, w, h]) => {
                let x0 = x.clamp(0.0, fw);
                let y0 = y.clamp(0.0, fh);
                let x1 = (x + w).clamp(0.0, fw);
                let y1 = (y + h).clamp(0.0, fh);
                (x0, y0, x1 - x0, y1 - y0)
            }
            None => (0.0, 0.0, fw, fh),
        };
        if w < 1.0 || h < 1.0 {
            return Err(Error::config("snapshot: empty region"));
        }
        let (ow, oh) = match (opts.width, opts.height) {
            (Some(ow), Some(oh)) => (ow, oh),
            (Some(ow), None) => (ow, ((h * ow as f32 / w).round() as u32).max(1)),
            (None, Some(oh)) => (((w * oh as f32 / h).round() as u32).max(1), oh),
            (None, None) => (w.round() as u32, h.round() as u32),
        };
        if ow > 16384 || oh > 16384 || ow > 65535 || oh > 65535 {
            return Err(Error::config("snapshot: output too large"));
        }
        let spec = TensorSpec {
            width: ow,
            height: oh,
            dtype: camtensor_core::DType::U8,
            layout: camtensor_core::Layout::Nhwc,
            color: camtensor_core::ColorOrder::Rgb,
            resize: camtensor_core::ResizeMode::Stretch,
            ..Default::default()
        };
        let rgb = self.crop(frame, &[Roi::Box { x, y, w, h }], &spec, Some(Device::Cpu))?;
        let bytes =
            rgb.as_bytes().ok_or_else(|| Error::decode("snapshot: expected host pixels"))?;
        let mut out = Vec::with_capacity(bytes.len() / 8);
        jpeg_encoder::Encoder::new(&mut out, opts.quality.clamp(1, 100))
            .encode(bytes, ow as u16, oh as u16, jpeg_encoder::ColorType::Rgb)
            .map_err(|e| Error::decode(format!("jpeg encode: {e}")))?;
        Ok(out)
    }

    /// Copies a tensor (host or device) into a host buffer of exactly
    /// `tensor.byte_len()` bytes.
    pub fn copy_to_host(&self, tensor: &Tensor, dst: &mut [u8]) -> Result<()> {
        if dst.len() != tensor.byte_len() {
            return Err(Error::config("copy_to_host: destination size mismatch"));
        }
        if let Some(b) = tensor.as_bytes() {
            dst.copy_from_slice(b);
            return Ok(());
        }
        let mut ctx = self.backend_for(tensor.device()).worker()?;
        ctx.download(tensor.data_ptr(), dst)
    }

    /// Copies a tensor to a new host `Vec<u8>`.
    pub fn to_host_vec(&self, tensor: &Tensor) -> Result<Vec<u8>> {
        let mut v = vec![0u8; tensor.byte_len()];
        self.copy_to_host(tensor, &mut v)?;
        Ok(v)
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
        EngineStats {
            device: self.device().to_string(),
            devices: self.devices().iter().map(ToString::to_string).collect(),
            streams,
            outputs,
        }
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
        // Close outputs first: writers blocked on a full output (`block`
        // backpressure) wake up, so stopping the streams cannot deadlock.
        for o in self.inner.outputs.read().values() {
            o.close();
        }
        let streams: Vec<StreamHandle> =
            self.inner.streams.lock().drain().map(|(_, h)| h).collect();
        drop(streams);
    }

    pub(crate) fn runtime(&self) -> &tokio::runtime::Handle {
        self.inner.rt.handle()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        for o in self.outputs.get_mut().values() {
            o.close();
        }
        self.streams.get_mut().clear();
    }
}

/// Options for [`Engine::snapshot`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotOptions {
    /// Region `[x, y, w, h]` in frame pixels (default: the whole frame).
    pub region: Option<[f32; 4]>,
    /// Output width (default: region width, or scaled with `height`).
    pub width: Option<u32>,
    /// Output height (default: region height, or scaled with `width`).
    pub height: Option<u32>,
    /// JPEG quality, 1..=100.
    pub quality: u8,
}

impl Default for SnapshotOptions {
    fn default() -> Self {
        Self { region: None, width: None, height: None, quality: 90 }
    }
}

/// Blocking iterator over an output's batches.
pub struct Subscription {
    group: Arc<OutputGroup>,
    device: Option<Device>,
}

impl std::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription").field("device", &self.device).finish_non_exhaustive()
    }
}

impl Subscription {
    /// Waits for the next batch with a timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Batch> {
        self.recv(Some(timeout))
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Batch> {
        match self.device {
            Some(d) => self.group.shard_for(d)?.recv(timeout),
            None => self.group.recv(timeout),
        }
    }
}

impl Iterator for Subscription {
    type Item = Batch;
    fn next(&mut self) -> Option<Batch> {
        self.recv(None).ok()
    }
}

/// Expands the configured device list into one backend per distinct device.
fn resolve_backends(cfg: &EngineConfig) -> Result<Vec<Backend>> {
    let list = if cfg.devices.is_empty() { vec![cfg.device] } else { cfg.devices.clone() };
    let mut devices: Vec<Device> = Vec::new();
    for d in list {
        let expanded = match d {
            // `auto` in a device list: every CUDA device (the CPU if none).
            Device::Auto if !cfg.devices.is_empty() => all_cuda_devices(),
            other => vec![other],
        };
        for e in expanded {
            if !devices.contains(&e) {
                devices.push(e);
            }
        }
    }
    let mut backends: Vec<Backend> = Vec::new();
    for d in devices {
        let b = Backend::resolve(d)?;
        // Distinct requested devices may resolve to the same one (auto -> cuda:0).
        if !backends.iter().any(|x| x.device() == b.device()) {
            backends.push(b);
        }
    }
    if backends.is_empty() {
        return Err(Error::config("no device configured"));
    }
    Ok(backends)
}

fn all_cuda_devices() -> Vec<Device> {
    #[cfg(feature = "cuda")]
    {
        let n = if camtensor_cuda::is_available() { camtensor_cuda::device_count() } else { 0 };
        if n > 0 {
            return (0..n as u32).map(Device::Cuda).collect();
        }
    }
    vec![Device::Cpu]
}
