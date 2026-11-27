//! CUDA devices and their primary contexts.

use std::collections::HashMap;
use std::ffi::{c_char, c_int};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use camtensor_core::{Error, Result};

use crate::error::{check, driver_api};
use crate::sys::driver::{self, CUcontext, CUdevice};

/// A CUDA device with its retained primary context.
///
/// Obtained through [`CudaDevice::get`], which caches one instance per
/// ordinal for the lifetime of the process (retaining and releasing the
/// primary context repeatedly would tear down every allocation and module).
///
/// All camtensor objects use the device's *primary* context, which is the
/// same context the CUDA runtime (PyTorch, TensorRT, ONNX Runtime) uses, so
/// device pointers can be shared with those libraries directly.
pub struct CudaDevice {
    ordinal: u32,
    dev: CUdevice,
    ctx: CUcontext,
    name: String,
    compute_capability: (u32, u32),
    total_memory: usize,
    multiprocessors: u32,
}

// SAFETY: a CUDA context handle may be used from any thread (it is made
// current per call through `bind`); the remaining fields are plain data.
unsafe impl Send for CudaDevice {}
// SAFETY: see above; `CudaDevice` has no interior mutability.
unsafe impl Sync for CudaDevice {}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("ordinal", &self.ordinal)
            .field("name", &self.name)
            .field("compute_capability", &self.compute_capability)
            .field("total_memory", &self.total_memory)
            .finish()
    }
}

static DEVICES: OnceLock<Mutex<HashMap<u32, Arc<CudaDevice>>>> = OnceLock::new();

/// Number of CUDA devices (0 when the driver is missing).
pub fn device_count() -> usize {
    let Ok(api) = driver::api() else { return 0 };
    let mut n: c_int = 0;
    // SAFETY: cuInit succeeded inside `driver::api`; `n` is a valid out pointer.
    let r = unsafe { (api.cuDeviceGetCount)(&mut n) };
    if r == driver::CUDA_SUCCESS { n.max(0) as usize } else { 0 }
}

impl CudaDevice {
    /// Returns the (cached) device for `ordinal`.
    pub fn get(ordinal: u32) -> Result<Arc<CudaDevice>> {
        let cache = DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(d) = map.get(&ordinal) {
            return Ok(d.clone());
        }
        let dev = Arc::new(Self::open(ordinal)?);
        map.insert(ordinal, dev.clone());
        Ok(dev)
    }

    fn open(ordinal: u32) -> Result<Self> {
        let api = driver_api()?;
        let count = device_count();
        if ordinal as usize >= count {
            return Err(Error::unavailable(format!(
                "CUDA device {ordinal} not found ({count} device(s) present)"
            )));
        }
        let mut dev: CUdevice = 0;
        // SAFETY: valid out pointer, ordinal checked against the count.
        check(unsafe { (api.cuDeviceGet)(&mut dev, ordinal as c_int) }, "cuDeviceGet")?;

        let mut name_buf = [0 as c_char; 256];
        check(
            // SAFETY: the buffer is 256 bytes and the driver writes at most `len`.
            unsafe { (api.cuDeviceGetName)(name_buf.as_mut_ptr(), name_buf.len() as c_int, dev) },
            "cuDeviceGetName",
        )?;
        // SAFETY: cuDeviceGetName NUL-terminates within the buffer.
        let name =
            unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }.to_string_lossy().into_owned();

        let attr = |a: c_int, what: &str| -> Result<i32> {
            let mut v: c_int = 0;
            // SAFETY: valid out pointer and device handle.
            check(unsafe { (api.cuDeviceGetAttribute)(&mut v, a, dev) }, what)?;
            Ok(v)
        };
        let major =
            attr(driver::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "cuDeviceGetAttribute")?;
        let minor =
            attr(driver::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "cuDeviceGetAttribute")?;
        let sms = attr(driver::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "cuDeviceGetAttribute")?;

        let mut total: usize = 0;
        // SAFETY: valid out pointer and device handle.
        check(unsafe { (api.cuDeviceTotalMem)(&mut total, dev) }, "cuDeviceTotalMem")?;

        let mut ctx: CUcontext = std::ptr::null_mut();
        check(
            // SAFETY: valid out pointer and device handle; released in Drop.
            unsafe { (api.cuDevicePrimaryCtxRetain)(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;

        Ok(Self {
            ordinal,
            dev,
            ctx,
            name,
            compute_capability: (major as u32, minor as u32),
            total_memory: total,
            multiprocessors: sms as u32,
        })
    }

    /// Makes this device's context current on the calling thread until the
    /// returned guard is dropped (push/pop, so nesting is fine).
    pub fn bind(&self) -> Result<ContextGuard<'_>> {
        let api = driver_api()?;
        // SAFETY: `ctx` is a retained primary context, valid while `self` lives.
        check(unsafe { (api.cuCtxPushCurrent)(self.ctx) }, "cuCtxPushCurrent")?;
        Ok(ContextGuard { _device: PhantomData, _not_send: PhantomData })
    }

    /// Blocks until all work on the device has completed.
    pub fn synchronize(&self) -> Result<()> {
        let api = driver_api()?;
        let _g = self.bind()?;
        // SAFETY: a context is current.
        check(unsafe { (api.cuCtxSynchronize)() }, "cuCtxSynchronize")
    }

    /// Blocks until work queued on the legacy default (NULL) stream has
    /// completed. Some driver components (e.g. NVDEC JPEG post-processing)
    /// enqueue there regardless of the stream they are given, and
    /// non-blocking streams do not order against it.
    pub fn synchronize_default_stream(&self) -> Result<()> {
        let api = driver_api()?;
        let _g = self.bind()?;
        // SAFETY: the NULL stream is always valid while a context is current.
        check(unsafe { (api.cuStreamSynchronize)(std::ptr::null_mut()) }, "cuStreamSynchronize")
    }

    /// `(free, total)` device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let api = driver_api()?;
        let _g = self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: a context is current; valid out pointers.
        check(unsafe { (api.cuMemGetInfo)(&mut free, &mut total) }, "cuMemGetInfo")?;
        Ok((free, total))
    }

    /// Device ordinal.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Marketing name (e.g. "NVIDIA GeForce RTX 4090").
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Compute capability `(major, minor)`.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Total device memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.total_memory
    }

    /// Number of streaming multiprocessors.
    pub fn multiprocessors(&self) -> u32 {
        self.multiprocessors
    }

    /// Raw `CUcontext` (the primary context).
    pub fn raw_context(&self) -> CUcontext {
        self.ctx
    }

    /// Raw `CUdevice`.
    pub fn raw_device(&self) -> CUdevice {
        self.dev
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if let Ok(api) = driver::api() {
            // SAFETY: balances the retain in `open`.
            unsafe { (api.cuDevicePrimaryCtxRelease)(self.dev) };
        }
    }
}

/// Keeps a CUDA context current on this thread; pops it on drop.
///
/// Not `Send`: the context stack is per thread.
#[derive(Debug)]
pub struct ContextGuard<'a> {
    _device: PhantomData<&'a CudaDevice>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        if let Ok(api) = driver::api() {
            let mut old: CUcontext = std::ptr::null_mut();
            // SAFETY: balances the push in `CudaDevice::bind` on this thread.
            unsafe { (api.cuCtxPopCurrent)(&mut old) };
        }
    }
}
