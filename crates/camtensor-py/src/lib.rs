//! Python bindings for camtensor (`import camtensor`).
//!
//! Tensors are exported zero-copy through DLPack (`torch.from_dlpack`,
//! `numpy.from_dlpack`, `cupy.from_dlpack`, ...), `__cuda_array_interface__`
//! and, for host tensors, the NumPy `__array_interface__`.

use std::time::Duration;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use camtensor::{
    Backpressure, BatchConfig, DType, Device, Engine as RsEngine, EngineConfig, Error, Frame,
    GatingConfig, GatingMode, OutputConfig, Roi, StreamConfig, Tensor as RsTensor, TensorSpec,
};

mod dlpack;

fn to_py(e: Error) -> PyErr {
    match e {
        Error::Timeout => PyTimeoutError::new_err("timed out"),
        Error::NotFound(m) => PyKeyError::new_err(m),
        Error::Config(m) => PyValueError::new_err(m),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn from_py_json<T: serde::de::DeserializeOwned>(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<T> {
    let json = obj.py().import("json")?.call_method1("dumps", (obj,))?;
    let s: String = json.extract()?;
    serde_norway::from_str(&s).map_err(|e| PyValueError::new_err(format!("invalid {what}: {e}")))
}

fn json_to_py<'py>(py: Python<'py>, v: &impl serde::Serialize) -> PyResult<Bound<'py, PyAny>> {
    let s = serde_json::to_string(v).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    py.import("json")?.call_method1("loads", (s,))
}

/// A camtensor engine: RTSP streams in, batches of tensors out.
#[pyclass(module = "camtensor._camtensor", name = "Engine", frozen)]
struct PyEngine {
    inner: RsEngine,
}

#[pymethods]
impl PyEngine {
    /// Create an engine.
    ///
    /// `config` may be `None`, a dict, a YAML/JSON string, or a path to a
    /// YAML/JSON file. `device` ("auto", "cpu", "cuda:0") overrides the
    /// configured device.
    #[new]
    #[pyo3(signature = (config=None, device=None))]
    fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let mut cfg = match config {
            None => EngineConfig::default(),
            Some(obj) if obj.is_instance_of::<PyDict>() => {
                let c: EngineConfig = from_py_json(obj, "config")?;
                c
            }
            Some(obj) => {
                let s: String = obj.extract()?;
                let path = std::path::Path::new(&s);
                if !s.contains('\n') && path.is_file() {
                    EngineConfig::from_path(path).map_err(to_py)?
                } else {
                    EngineConfig::from_yaml(&s).map_err(to_py)?
                }
            }
        };
        if let Some(d) = device {
            cfg.device = d.parse().map_err(to_py)?;
        }
        let inner = py.detach(|| RsEngine::new(cfg)).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Device used for decoding and default outputs ("cpu" or "cuda:N").
    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    /// Names of the outputs.
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.inner.outputs()
    }

    /// Ids of the running streams.
    #[getter]
    fn streams(&self) -> Vec<String> {
        self.inner.streams()
    }

    /// Add a named output (a tensor format + batching policy).
    #[pyo3(signature = (
        name, width=640, height=640, *, dtype="f32", layout="nchw", color="rgb",
        resize="letterbox", interp="bilinear", normalize=None, pad=None,
        batch=1, max_wait_ms=0, pad_to_max=false, keep_frame=false, max_fps=None,
        backpressure="drop_oldest", queue_depth=4, device=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_output(
        &self,
        name: &str,
        width: u32,
        height: u32,
        dtype: &str,
        layout: &str,
        color: &str,
        resize: &str,
        interp: &str,
        normalize: Option<&Bound<'_, PyAny>>,
        pad: Option<&Bound<'_, PyAny>>,
        batch: usize,
        max_wait_ms: u64,
        pad_to_max: bool,
        keep_frame: bool,
        max_fps: Option<f64>,
        backpressure: &str,
        queue_depth: usize,
        device: Option<&str>,
    ) -> PyResult<()> {
        let tensor =
            build_spec(width, height, dtype, layout, color, resize, interp, normalize, pad)?;
        let cfg = OutputConfig {
            tensor,
            batch: BatchConfig { max_size: batch, max_wait_ms, pad_to_max },
            max_fps,
            keep_frame,
            backpressure: parse_enum::<Backpressure>(backpressure, "backpressure")?,
            queue_depth,
            device: device.map(|d| d.parse::<Device>()).transpose().map_err(to_py)?,
        };
        self.inner.add_output(name, cfg).map_err(to_py)
    }

    /// Start consuming an RTSP stream.
    #[pyo3(signature = (id, url, outputs, *, gating=None, transport="tcp", decoder=None, username=None, password=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_stream(
        &self,
        py: Python<'_>,
        id: &str,
        url: &str,
        outputs: Vec<String>,
        gating: Option<&Bound<'_, PyAny>>,
        transport: &str,
        decoder: Option<&str>,
        username: Option<String>,
        password: Option<String>,
    ) -> PyResult<()> {
        let gating = match gating {
            None => GatingConfig::default(),
            Some(g) if g.extract::<String>().is_ok() => GatingConfig {
                mode: parse_enum::<GatingMode>(&g.extract::<String>()?, "gating")?,
                ..Default::default()
            },
            Some(g) => from_py_json(g, "gating")?,
        };
        let cfg = StreamConfig {
            id: id.to_string(),
            url: url.to_string(),
            username,
            password,
            transport: parse_enum(transport, "transport")?,
            outputs,
            gating,
            decoder: decoder.map(|d| parse_enum(d, "decoder")).transpose()?,
            ..Default::default()
        };
        py.detach(|| self.inner.add_stream(cfg)).map_err(to_py)
    }

    /// Stop and remove a stream.
    fn remove_stream(&self, py: Python<'_>, id: &str) -> PyResult<()> {
        py.detach(|| self.inner.remove_stream(id)).map_err(to_py)
    }

    /// Bypass gating (decode everything) on a stream for `seconds`.
    fn force_active(&self, id: &str, seconds: f64) -> PyResult<()> {
        self.inner.force_active(id, Duration::from_secs_f64(seconds.max(0.0))).map_err(to_py)
    }

    /// Wait for the next batch of `output`. Raises `TimeoutError` after
    /// `timeout` seconds (None waits forever, releasing the GIL).
    #[pyo3(signature = (output, timeout=None))]
    fn recv(&self, py: Python<'_>, output: &str, timeout: Option<f64>) -> PyResult<PyBatch> {
        let deadline =
            timeout.map(|s| std::time::Instant::now() + Duration::from_secs_f64(s.max(0.0)));
        loop {
            // Wake up periodically so Ctrl+C (KeyboardInterrupt) works.
            let mut wait = Duration::from_millis(200);
            if let Some(d) = deadline {
                wait = wait.min(d.saturating_duration_since(std::time::Instant::now()));
            }
            match py.detach(|| self.inner.recv(output, Some(wait))) {
                Ok(b) => return Ok(PyBatch::from(b)),
                Err(Error::Timeout) => {
                    py.check_signals()?;
                    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                        return Err(PyTimeoutError::new_err("no batch within timeout"));
                    }
                }
                Err(e) => return Err(to_py(e)),
            }
        }
    }

    /// Return the next batch if one is ready, else None.
    fn try_recv(&self, output: &str) -> PyResult<Option<PyBatch>> {
        Ok(self.inner.try_recv(output).map_err(to_py)?.map(PyBatch::from))
    }

    /// Iterate over the batches of `output` until the engine is closed.
    fn batches(slf: Py<Self>, output: String) -> PyBatchIter {
        PyBatchIter { engine: slf, output }
    }

    /// Extract crops of a kept frame as a tensor `[len(rois), ...]`.
    ///
    /// Each ROI is one of: `(x, y, w, h)` box, a dict `{"kind": "box", ...}`,
    /// `{"kind": "affine", "matrix": [6 floats]}` (OpenCV convention),
    /// `{"kind": "landmarks", "points": [[x, y] x 5]}`, or a 5x2 list of
    /// landmarks (ArcFace alignment).
    #[pyo3(signature = (
        frame, rois, width=112, height=112, *, dtype="f32", layout="nchw", color="rgb",
        resize="stretch", interp="bilinear", normalize=None, pad=None, device=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn crop(
        &self,
        py: Python<'_>,
        frame: &PyFrame,
        rois: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        dtype: &str,
        layout: &str,
        color: &str,
        resize: &str,
        interp: &str,
        normalize: Option<&Bound<'_, PyAny>>,
        pad: Option<&Bound<'_, PyAny>>,
        device: Option<&str>,
    ) -> PyResult<PyTensor> {
        let spec = build_spec(width, height, dtype, layout, color, resize, interp, normalize, pad)?;
        let rois = parse_rois(rois)?;
        let device = device.map(|d| d.parse::<Device>()).transpose().map_err(to_py)?;
        let t = py.detach(|| self.inner.crop(&frame.inner, &rois, &spec, device)).map_err(to_py)?;
        Ok(PyTensor { inner: t })
    }

    /// Runtime statistics as a dict.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.stats())
    }

    /// Runtime statistics in Prometheus text format.
    fn prometheus(&self) -> String {
        self.inner.stats().to_prometheus()
    }

    /// Current configuration as a dict.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.config())
    }

    /// Stop every stream and wake up consumers.
    fn close(&self, py: Python<'_>) {
        py.detach(|| self.inner.shutdown());
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyTuple>) {
        self.close(py);
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(device={}, outputs={:?}, streams={:?})",
            self.inner.device(),
            self.inner.outputs(),
            self.inner.streams()
        )
    }
}

#[pyclass(module = "camtensor._camtensor", name = "BatchIterator")]
struct PyBatchIter {
    engine: Py<PyEngine>,
    output: String,
}

#[pymethods]
impl PyBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyBatch>> {
        let engine = self.engine.get();
        loop {
            match py.detach(|| engine.inner.recv(&self.output, Some(Duration::from_millis(200)))) {
                Ok(b) => return Ok(Some(PyBatch::from(b))),
                Err(Error::Timeout) => py.check_signals()?,
                Err(Error::Closed) => return Ok(None),
                Err(e) => return Err(to_py(e)),
            }
        }
    }
}

/// A batch of preprocessed images. Supports `torch.from_dlpack(batch)`.
#[pyclass(module = "camtensor._camtensor", name = "Batch", frozen)]
struct PyBatch {
    inner: camtensor::Batch,
}

impl From<camtensor::Batch> for PyBatch {
    fn from(inner: camtensor::Batch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBatch {
    /// The batch tensor.
    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor { inner: self.inner.tensor().clone() }
    }

    /// Output name.
    #[getter]
    fn output(&self) -> &str {
        self.inner.output()
    }

    /// Batch counter.
    #[getter]
    fn seq(&self) -> u64 {
        self.inner.seq()
    }

    /// Per-slot metadata.
    #[getter]
    fn items(&self) -> Vec<PyItem> {
        self.inner.items().iter().map(|i| PyItem { inner: i.clone() }).collect()
    }

    /// Number of valid (non-padding) slots.
    #[getter]
    fn valid_count(&self) -> usize {
        self.inner.valid_count()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, i: isize) -> PyResult<PyItem> {
        let n = self.inner.len() as isize;
        let idx = if i < 0 { n + i } else { i };
        if idx < 0 || idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err("batch index out of range"));
        }
        Ok(PyItem { inner: self.inner.items()[idx as usize].clone() })
    }

    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<&Bound<'py, PyAny>>,
        dl_device: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.tensor().__dlpack__(py, stream, max_version, dl_device, copy)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        self.tensor().__dlpack_device__()
    }

    /// Copy (or view, for host tensors) as a NumPy array.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tensor().numpy(py)
    }

    fn __repr__(&self) -> String {
        let t = self.inner.tensor();
        format!(
            "Batch(output={:?}, len={}, valid={}, shape={:?}, dtype={:?}, device={})",
            self.inner.output(),
            self.inner.len(),
            self.inner.valid_count(),
            t.shape(),
            t.dtype(),
            t.device()
        )
    }
}

/// Metadata of one batch slot.
#[pyclass(module = "camtensor._camtensor", name = "Item", frozen)]
struct PyItem {
    inner: camtensor::BatchItem,
}

#[pymethods]
impl PyItem {
    #[getter]
    fn stream_id(&self) -> &str {
        &self.inner.meta.stream_id
    }
    #[getter]
    fn valid(&self) -> bool {
        self.inner.valid
    }
    #[getter]
    fn seq(&self) -> u64 {
        self.inner.meta.seq
    }
    #[getter]
    fn pts(&self) -> i64 {
        self.inner.meta.pts
    }
    #[getter]
    fn keyframe(&self) -> bool {
        self.inner.meta.keyframe
    }
    #[getter]
    fn src_width(&self) -> u32 {
        self.inner.meta.src_width
    }
    #[getter]
    fn src_height(&self) -> u32 {
        self.inner.meta.src_height
    }
    /// Continuous dst_from_src affine `(a, b, c, d, e, f)`.
    #[getter]
    fn transform(&self) -> [f32; 6] {
        self.inner.meta.transform.0
    }
    #[getter]
    fn capture_time_ns(&self) -> Option<u64> {
        self.inner.meta.capture_time_ns
    }
    #[getter]
    fn recv_time_ns(&self) -> u64 {
        self.inner.meta.recv_time_ns
    }
    #[getter]
    fn ready_time_ns(&self) -> u64 {
        self.inner.meta.ready_time_ns
    }
    #[getter]
    fn latency_ms(&self) -> f64 {
        self.inner.meta.latency_ms()
    }
    #[getter]
    fn dropped_since_last(&self) -> u64 {
        self.inner.meta.dropped_since_last
    }
    /// The full-resolution frame (outputs with `keep_frame=True`).
    #[getter]
    fn frame(&self) -> Option<PyFrame> {
        self.inner.frame.clone().map(|inner| PyFrame { inner })
    }
    /// Map a point from tensor pixels to source pixels.
    fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        self.inner.meta.to_source(x, y)
    }
    /// Map a box `(x1, y1, x2, y2)` from tensor pixels to source pixels.
    fn box_to_source(&self, b: [f32; 4]) -> [f32; 4] {
        self.inner.meta.box_to_source(b)
    }
    /// Map many boxes (list of `(x1, y1, x2, y2)`).
    fn boxes_to_source(&self, boxes: Vec<[f32; 4]>) -> Vec<[f32; 4]> {
        boxes.into_iter().map(|b| self.inner.meta.box_to_source(b)).collect()
    }
    fn __repr__(&self) -> String {
        let m = &self.inner.meta;
        format!(
            "Item(stream={:?}, seq={}, pts={}, src={}x{}, valid={}, latency_ms={:.2})",
            &*m.stream_id,
            m.seq,
            m.pts,
            m.src_width,
            m.src_height,
            self.inner.valid,
            m.latency_ms()
        )
    }
}

/// A decoded full-resolution frame (opaque; pass it to `Engine.crop`).
#[pyclass(module = "camtensor._camtensor", name = "Frame", frozen)]
struct PyFrame {
    inner: Frame,
}

#[pymethods]
impl PyFrame {
    #[getter]
    fn width(&self) -> u32 {
        self.inner.desc.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.desc.height
    }
    #[getter]
    fn format(&self) -> String {
        format!("{:?}", self.inner.desc.format).to_lowercase()
    }
    #[getter]
    fn device(&self) -> &'static str {
        if self.inner.is_device() { "cuda" } else { "cpu" }
    }
    #[getter]
    fn pts(&self) -> i64 {
        self.inner.pts
    }
    fn __repr__(&self) -> String {
        format!("Frame({}x{}, {}, {})", self.width(), self.height(), self.format(), self.device())
    }
}

/// A dense tensor on the CPU or a CUDA device (zero-copy via DLPack).
#[pyclass(module = "camtensor._camtensor", name = "Tensor", frozen)]
struct PyTensor {
    inner: RsTensor,
}

fn dtype_str(d: DType) -> &'static str {
    match d {
        DType::U8 => "uint8",
        DType::F16 => "float16",
        DType::F32 => "float32",
    }
}

fn typestr(d: DType) -> &'static str {
    match d {
        DType::U8 => "|u1",
        DType::F16 => "<f2",
        DType::F32 => "<f4",
    }
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_str(self.inner.dtype())
    }
    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }
    #[getter]
    fn data_ptr(&self) -> u64 {
        self.inner.data_ptr()
    }
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.byte_len()
    }

    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<&Bound<'py, PyAny>>,
        dl_device: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Tensors are fully written (stream-synchronised) before delivery,
        // so the consumer stream needs no extra synchronisation.
        let _ = (stream, dl_device);
        if copy == Some(true) {
            return Err(pyo3::exceptions::PyBufferError::new_err("copy=True is not supported"));
        }
        let versioned = match max_version {
            Some(v) if !v.is_none() => {
                let (major, _minor): (u32, u32) = v.extract()?;
                major >= 1
            }
            _ => false,
        };
        dlpack::export(py, &self.inner, versioned)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        dlpack::device(&self.inner)
    }

    /// NumPy array interface (host tensors only).
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.inner.device().is_cuda() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "device tensor: use __cuda_array_interface__ or DLPack",
            ));
        }
        self.interface(py, 3)
    }

    /// CUDA array interface (device tensors only; Numba, CuPy).
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.inner.device().is_cuda() {
            return Err(pyo3::exceptions::PyAttributeError::new_err("host tensor"));
        }
        let d = self.interface(py, 3)?;
        d.set_item("stream", py.None())?;
        Ok(d)
    }

    /// View (host) or copy (device, needs the optional CPU fallback) as NumPy.
    fn numpy<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if slf.get().inner.device().is_cuda() {
            return Err(PyRuntimeError::new_err(
                "device tensor: use torch.from_dlpack / cupy.from_dlpack, or an output with device='cpu'",
            ));
        }
        py.import("numpy")?.call_method1("asarray", (slf,))
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={})",
            self.inner.shape(),
            dtype_str(self.inner.dtype()),
            self.inner.device()
        )
    }
}

impl PyTensor {
    fn interface<'py>(&self, py: Python<'py>, version: u32) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", PyTuple::new(py, self.inner.shape())?)?;
        d.set_item("typestr", typestr(self.inner.dtype()))?;
        d.set_item("data", (self.inner.data_ptr(), false))?;
        d.set_item("version", version)?;
        d.set_item("strides", py.None())?;
        Ok(d)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_spec(
    width: u32,
    height: u32,
    dtype: &str,
    layout: &str,
    color: &str,
    resize: &str,
    interp: &str,
    normalize: Option<&Bound<'_, PyAny>>,
    pad: Option<&Bound<'_, PyAny>>,
) -> PyResult<TensorSpec> {
    let mut spec = TensorSpec {
        width,
        height,
        dtype: parse_enum(dtype, "dtype")?,
        layout: parse_enum(layout, "layout")?,
        color: parse_enum(color, "color")?,
        resize: parse_enum(resize, "resize")?,
        interp: parse_enum(interp, "interp")?,
        ..Default::default()
    };
    if let Some(n) = normalize {
        spec.normalize = from_py_json(n, "normalize")?;
    }
    if let Some(p) = pad {
        spec.pad = match p.extract::<f32>() {
            Ok(v) => [v; 3],
            Err(_) => p.extract::<[f32; 3]>()?,
        };
    }
    spec.validate().map_err(to_py)?;
    Ok(spec)
}

fn parse_enum<T: serde::de::DeserializeOwned>(s: &str, what: &str) -> PyResult<T> {
    let norm = match s.to_ascii_lowercase().as_str() {
        "float32" => "f32".to_string(),
        "float16" | "half" => "f16".to_string(),
        "uint8" => "u8".to_string(),
        "nvdec" | "gpu" if what == "decoder" => "nvdec".to_string(),
        other => other.replace('-', "_"),
    };
    serde_norway::from_str::<T>(&norm)
        .map_err(|_| PyValueError::new_err(format!("invalid {what}: {s:?}")))
}

fn parse_rois(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Roi>> {
    let list = obj.cast::<PyList>().map_err(|_| PyValueError::new_err("rois must be a list"))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        if item.is_instance_of::<PyDict>() {
            out.push(from_py_json::<Roi>(&item, "roi")?);
        } else if let Ok(b) = item.extract::<[f32; 4]>() {
            out.push(Roi::Box { x: b[0], y: b[1], w: b[2], h: b[3] });
        } else if let Ok(points) = item.extract::<[[f32; 2]; 5]>() {
            out.push(Roi::Landmarks { points });
        } else if let Ok(m) = item.extract::<[f32; 6]>() {
            out.push(Roi::Affine { matrix: m });
        } else {
            return Err(PyValueError::new_err(
                "roi must be (x, y, w, h), 5x2 landmarks, 6 affine values or a dict",
            ));
        }
    }
    Ok(out)
}

/// Version of the native extension.
#[pyfunction]
fn version() -> &'static str {
    camtensor::VERSION
}

/// Which optional backends are available on this machine.
#[pyfunction]
fn capabilities<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &camtensor::capabilities())
}

/// Python module definition.
#[pymodule]
fn _camtensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyBatch>()?;
    m.add_class::<PyBatchIter>()?;
    m.add_class::<PyItem>()?;
    m.add_class::<PyFrame>()?;
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(capabilities, m)?)?;
    m.add("__version__", camtensor::VERSION)?;
    Ok(())
}
