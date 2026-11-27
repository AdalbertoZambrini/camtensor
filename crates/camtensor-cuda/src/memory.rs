//! Device and pinned host memory, plus a small reusable buffer pool.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use camtensor_core::{DeviceMemory, Error, Result};

use crate::device::CudaDevice;
use crate::error::{check, driver_api};
use crate::sys::driver::{self, CUdeviceptr};

/// An owned allocation of device memory (`cuMemAlloc`).
///
/// Freeing (`Drop`) goes through `cuMemFree`, which waits for outstanding
/// work that may still use the memory, so dropping a buffer while kernels
/// are queued is safe (but stalls).
pub struct DeviceBuffer {
    ptr: CUdeviceptr,
    len: usize,
    device: Arc<CudaDevice>,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("device", &self.device.ordinal())
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("len", &self.len)
            .finish()
    }
}

impl DeviceBuffer {
    /// Allocates `len` bytes of uninitialised device memory.
    pub fn alloc(device: &Arc<CudaDevice>, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self { ptr: 0, len: 0, device: device.clone() });
        }
        let api = driver_api()?;
        let _g = device.bind()?;
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: context current; valid out pointer.
        check(unsafe { (api.cuMemAlloc)(&mut ptr, len) }, "cuMemAlloc")?;
        Ok(Self { ptr, len, device: device.clone() })
    }

    /// Allocates and uploads `data` (synchronous).
    pub fn from_host(device: &Arc<CudaDevice>, data: &[u8]) -> Result<Self> {
        let b = Self::alloc(device, data.len())?;
        b.copy_from_host(0, data)?;
        Ok(b)
    }

    /// Allocates `len` zeroed bytes (synchronous).
    pub fn zeroed(device: &Arc<CudaDevice>, len: usize) -> Result<Self> {
        let b = Self::alloc(device, len)?;
        if len > 0 {
            let api = driver_api()?;
            let _g = device.bind()?;
            // SAFETY: the range is our allocation; null stream = legacy
            // default stream, followed by a context sync below.
            check(
                unsafe { (api.cuMemsetD8Async)(b.ptr, 0, len, std::ptr::null_mut()) },
                "cuMemsetD8Async",
            )?;
            // SAFETY: context current.
            check(unsafe { (api.cuCtxSynchronize)() }, "cuCtxSynchronize")?;
        }
        Ok(b)
    }

    /// Synchronous upload into `self[offset..]`.
    ///
    /// Uses the legacy default stream, which does *not* order against
    /// non-blocking streams: synchronise streams that use this buffer first.
    pub fn copy_from_host(&self, offset: usize, data: &[u8]) -> Result<()> {
        self.check_range(offset, data.len())?;
        if data.is_empty() {
            return Ok(());
        }
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: the range is inside our allocation; `data` is valid for
        // the duration of this synchronous call.
        check(
            unsafe {
                (api.cuMemcpyHtoD)(
                    self.ptr + offset as u64,
                    data.as_ptr() as *const c_void,
                    data.len(),
                )
            },
            "cuMemcpyHtoD",
        )
    }

    /// Synchronous download of `self[offset..offset + out.len()]`.
    ///
    /// Same ordering caveat as [`DeviceBuffer::copy_from_host`].
    pub fn copy_to_host(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        self.check_range(offset, out.len())?;
        if out.is_empty() {
            return Ok(());
        }
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: range checked; `out` is valid for this synchronous call.
        check(
            unsafe {
                (api.cuMemcpyDtoH)(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr + offset as u64,
                    out.len(),
                )
            },
            "cuMemcpyDtoH",
        )
    }

    /// Downloads the whole buffer (synchronous).
    pub fn to_vec(&self) -> Result<Vec<u8>> {
        let mut v = vec![0u8; self.len];
        self.copy_to_host(0, &mut v)?;
        Ok(v)
    }

    /// Raw device pointer.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Size in bytes.
    pub fn byte_len(&self) -> usize {
        self.len
    }

    /// Owning device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    fn check_range(&self, offset: usize, n: usize) -> Result<()> {
        match offset.checked_add(n) {
            Some(end) if end <= self.len => Ok(()),
            _ => Err(Error::cuda(format!(
                "range {offset}..{offset}+{n} out of bounds for a {}-byte device buffer",
                self.len
            ))),
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        if let (Ok(api), Ok(_g)) = (driver::api(), self.device.bind()) {
            // SAFETY: we own the allocation and free it exactly once.
            unsafe { (api.cuMemFree)(self.ptr) };
        }
    }
}

impl DeviceMemory for DeviceBuffer {
    fn device_ptr(&self) -> u64 {
        self.ptr
    }
    fn byte_len(&self) -> usize {
        self.len
    }
    fn device_ordinal(&self) -> u32 {
        self.device.ordinal()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Page-locked host memory (`cuMemHostAlloc`), required for truly
/// asynchronous host/device copies. Zero-initialised.
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
    device: Arc<CudaDevice>,
}

// SAFETY: the buffer is plain host memory exclusively owned by this value.
unsafe impl Send for PinnedHostBuffer {}
// SAFETY: shared access only hands out `&[u8]`.
unsafe impl Sync for PinnedHostBuffer {}

impl std::fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostBuffer").field("len", &self.len).finish()
    }
}

impl PinnedHostBuffer {
    /// Allocates `len` zeroed bytes of pinned host memory.
    pub fn new(device: &Arc<CudaDevice>, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                device: device.clone(),
            });
        }
        let api = driver_api()?;
        let _g = device.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: context current; valid out pointer.
        check(
            unsafe { (api.cuMemHostAlloc)(&mut p, len, driver::CU_MEMHOSTALLOC_PORTABLE) },
            "cuMemHostAlloc",
        )?;
        let ptr = p as *mut u8;
        // SAFETY: `ptr` points to `len` writable bytes just allocated.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        Ok(Self { ptr, len, device: device.clone() })
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` for a zero-length buffer.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Mutable host pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Contents.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` initialised bytes (zeroed at
        // allocation) and uniquely owned.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Mutable contents.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, with exclusive access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        if let (Ok(api), Ok(_g)) = (driver::api(), self.device.bind()) {
            // SAFETY: allocated by cuMemHostAlloc, freed once.
            unsafe { (api.cuMemFreeHost)(self.ptr as *mut c_void) };
        }
    }
}

/// A pool of equally sized device buffers.
///
/// [`BufferPool::get`] hands out an `Arc<DeviceBuffer>`; a buffer is reused
/// once every clone handed out has been dropped (the pool is then the only
/// holder). Up to `capacity` buffers are retained; beyond that, buffers are
/// allocated unpooled and freed when dropped.
///
/// Consumers must make sure GPU work reading a buffer has completed (e.g.
/// synchronise their stream) before dropping their last reference, since
/// the buffer may be overwritten right after.
pub struct BufferPool {
    device: Arc<CudaDevice>,
    byte_len: usize,
    capacity: usize,
    buffers: Mutex<Vec<Arc<DeviceBuffer>>>,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("byte_len", &self.byte_len)
            .field("capacity", &self.capacity)
            .field("allocated", &self.allocated())
            .finish()
    }
}

impl BufferPool {
    /// Creates an empty pool of `byte_len`-byte buffers retaining at most
    /// `capacity` of them.
    pub fn new(device: &Arc<CudaDevice>, byte_len: usize, capacity: usize) -> Self {
        Self { device: device.clone(), byte_len, capacity, buffers: Mutex::new(Vec::new()) }
    }

    /// Returns a free buffer, allocating one if needed.
    pub fn get(&self) -> Result<Arc<DeviceBuffer>> {
        let mut bufs = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        // Only the pool holds a reference: nobody else can clone it while we
        // hold the lock, so handing it out is race-free.
        if let Some(b) = bufs.iter().find(|b| Arc::strong_count(b) == 1 && Arc::weak_count(b) == 0)
        {
            return Ok(b.clone());
        }
        let b = Arc::new(DeviceBuffer::alloc(&self.device, self.byte_len)?);
        if bufs.len() < self.capacity {
            bufs.push(b.clone());
        }
        Ok(b)
    }

    /// Size of every buffer.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Maximum number of retained buffers.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of retained buffers.
    pub fn allocated(&self) -> usize {
        self.buffers.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Number of retained buffers currently handed out.
    pub fn in_use(&self) -> usize {
        self.buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|b| Arc::strong_count(b) > 1 || Arc::weak_count(b) > 0)
            .count()
    }
}
