//! CUDA streams, events and asynchronous copies.

use std::ffi::c_void;
use std::sync::Arc;

use camtensor_core::{Error, Result};

use crate::device::CudaDevice;
use crate::error::{check, driver_api};
use crate::memory::{DeviceBuffer, PinnedHostBuffer};
use crate::sys::driver::{self, CUDA_MEMCPY2D, CUevent, CUstream};

/// A non-blocking CUDA stream (does not synchronise with the legacy default
/// stream).
pub struct CudaStream {
    raw: CUstream,
    device: Arc<CudaDevice>,
}

// SAFETY: stream handles are not bound to a thread; every call binds the
// owning context first.
unsafe impl Send for CudaStream {}
// SAFETY: the CUDA driver API is thread-safe for concurrent use of a stream.
unsafe impl Sync for CudaStream {}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("device", &self.device.ordinal())
            .field("raw", &self.raw)
            .finish()
    }
}

/// One 2-D copy between pitched regions (host or device).
#[derive(Debug, Clone, Copy)]
pub struct Copy2d {
    /// Source address (device pointer or host address).
    pub src: u64,
    /// `true` if `src` is a host address.
    pub src_host: bool,
    /// Source row pitch in bytes.
    pub src_pitch: usize,
    /// Destination address (device pointer or host address).
    pub dst: u64,
    /// `true` if `dst` is a host address.
    pub dst_host: bool,
    /// Destination row pitch in bytes.
    pub dst_pitch: usize,
    /// Bytes per row to copy.
    pub width_bytes: usize,
    /// Number of rows.
    pub height: usize,
}

impl CudaStream {
    /// Creates a non-blocking stream on `device`.
    pub fn new(device: &Arc<CudaDevice>) -> Result<Self> {
        let api = driver_api()?;
        let _g = device.bind()?;
        let mut raw: CUstream = std::ptr::null_mut();
        check(
            // SAFETY: a context is current; valid out pointer.
            unsafe { (api.cuStreamCreate)(&mut raw, driver::CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(Self { raw, device: device.clone() })
    }

    /// Device owning the stream.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Raw `CUstream` handle (for interop with TensorRT, PyTorch, ...).
    pub fn raw(&self) -> CUstream {
        self.raw
    }

    /// Blocks until all work queued on the stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: valid stream handle, context current.
        check(unsafe { (api.cuStreamSynchronize)(self.raw) }, "cuStreamSynchronize")
    }

    /// `true` when all queued work has completed.
    pub fn is_idle(&self) -> Result<bool> {
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: valid stream handle, context current.
        let r = unsafe { (api.cuStreamQuery)(self.raw) };
        match r {
            driver::CUDA_SUCCESS => Ok(true),
            driver::CUDA_ERROR_NOT_READY => Ok(false),
            _ => check(r, "cuStreamQuery").map(|_| false),
        }
    }

    /// Makes future work on this stream wait for `event`.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: valid handles, context current.
        check(unsafe { (api.cuStreamWaitEvent)(self.raw, event.raw, 0) }, "cuStreamWaitEvent")
    }

    /// Copies `src` into `dst[offset..]` and waits for completion.
    pub fn upload(&self, dst: &DeviceBuffer, offset: usize, src: &[u8]) -> Result<()> {
        check_range(dst.byte_len(), offset, src.len())?;
        // SAFETY: range checked; `src` outlives the copy because we
        // synchronise before returning.
        unsafe {
            self.memcpy_htod_async(dst.device_ptr() + offset as u64, src.as_ptr(), src.len())?
        };
        self.synchronize()
    }

    /// Copies `src[offset..offset + dst.len()]` into `dst` and waits for
    /// completion.
    pub fn download(&self, src: &DeviceBuffer, offset: usize, dst: &mut [u8]) -> Result<()> {
        check_range(src.byte_len(), offset, dst.len())?;
        // SAFETY: range checked; `dst` outlives the copy because we
        // synchronise before returning.
        unsafe {
            self.memcpy_dtoh_async(dst.as_mut_ptr(), src.device_ptr() + offset as u64, dst.len())?
        };
        self.synchronize()
    }

    /// Asynchronous upload from pinned memory.
    ///
    /// # Safety
    /// `src` must not be dropped or written until the stream has passed this
    /// point (e.g. [`CudaStream::synchronize`]).
    pub unsafe fn upload_pinned_async(
        &self,
        dst: &DeviceBuffer,
        offset: usize,
        src: &PinnedHostBuffer,
    ) -> Result<()> {
        check_range(dst.byte_len(), offset, src.len())?;
        // SAFETY: range checked; liveness is the caller's contract.
        unsafe { self.memcpy_htod_async(dst.device_ptr() + offset as u64, src.as_ptr(), src.len()) }
    }

    /// Asynchronous download into pinned memory.
    ///
    /// # Safety
    /// `dst` must not be dropped, read or written until the stream has
    /// passed this point.
    pub unsafe fn download_pinned_async(
        &self,
        src: &DeviceBuffer,
        offset: usize,
        dst: &mut PinnedHostBuffer,
    ) -> Result<()> {
        check_range(src.byte_len(), offset, dst.len())?;
        // SAFETY: range checked; liveness is the caller's contract.
        unsafe {
            self.memcpy_dtoh_async(dst.as_mut_ptr(), src.device_ptr() + offset as u64, dst.len())
        }
    }

    /// Asynchronous device-to-device copy (bounds checked).
    pub fn copy_dtod_async(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        len: usize,
    ) -> Result<()> {
        check_range(dst.byte_len(), dst_offset, len)?;
        check_range(src.byte_len(), src_offset, len)?;
        // SAFETY: both ranges are inside live allocations; freeing device
        // memory synchronises, so the buffers outlive the copy.
        unsafe {
            self.memcpy_dtod_async(
                dst.device_ptr() + dst_offset as u64,
                src.device_ptr() + src_offset as u64,
                len,
            )
        }
    }

    /// Asynchronously fills `dst[offset..offset + len]` with `value`.
    pub fn memset_async(
        &self,
        dst: &DeviceBuffer,
        offset: usize,
        len: usize,
        value: u8,
    ) -> Result<()> {
        check_range(dst.byte_len(), offset, len)?;
        let api = driver_api()?;
        let _g = self.device.bind()?;
        check(
            // SAFETY: range inside a live allocation.
            unsafe {
                (api.cuMemsetD8Async)(dst.device_ptr() + offset as u64, value, len, self.raw)
            },
            "cuMemsetD8Async",
        )
    }

    /// Raw asynchronous host-to-device copy.
    ///
    /// # Safety
    /// `dst..dst+len` must be valid device memory and `src..src+len` valid
    /// host memory that stays alive and unmodified until the copy completes.
    pub unsafe fn memcpy_htod_async(&self, dst: u64, src: *const u8, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let api = driver_api()?;
        let _g = self.device.bind()?;
        check(
            // SAFETY: the caller upholds the pointer contract.
            unsafe { (api.cuMemcpyHtoDAsync)(dst, src as *const c_void, len, self.raw) },
            "cuMemcpyHtoDAsync",
        )
    }

    /// Raw asynchronous device-to-host copy.
    ///
    /// # Safety
    /// `src..src+len` must be valid device memory and `dst..dst+len` valid
    /// host memory that stays alive and untouched until the copy completes.
    pub unsafe fn memcpy_dtoh_async(&self, dst: *mut u8, src: u64, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let api = driver_api()?;
        let _g = self.device.bind()?;
        check(
            // SAFETY: the caller upholds the pointer contract.
            unsafe { (api.cuMemcpyDtoHAsync)(dst as *mut c_void, src, len, self.raw) },
            "cuMemcpyDtoHAsync",
        )
    }

    /// Raw asynchronous device-to-device copy.
    ///
    /// # Safety
    /// Both ranges must be valid device memory until the copy completes.
    pub unsafe fn memcpy_dtod_async(&self, dst: u64, src: u64, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: the caller upholds the pointer contract.
        check(unsafe { (api.cuMemcpyDtoDAsync)(dst, src, len, self.raw) }, "cuMemcpyDtoDAsync")
    }

    /// Raw asynchronous 2-D copy.
    ///
    /// # Safety
    /// Source and destination regions (`pitch * (height - 1) + width_bytes`
    /// bytes from each base address) must be valid for the duration of the
    /// copy; host regions must stay alive until it completes.
    pub unsafe fn memcpy_2d_async(&self, c: &Copy2d) -> Result<()> {
        if c.width_bytes == 0 || c.height == 0 {
            return Ok(());
        }
        let api = driver_api()?;
        let _g = self.device.bind()?;
        let mut p = CUDA_MEMCPY2D {
            srcPitch: c.src_pitch,
            dstPitch: c.dst_pitch,
            WidthInBytes: c.width_bytes,
            Height: c.height,
            ..Default::default()
        };
        if c.src_host {
            p.srcMemoryType = driver::CU_MEMORYTYPE_HOST;
            p.srcHost = c.src as usize as *const c_void;
        } else {
            p.srcMemoryType = driver::CU_MEMORYTYPE_DEVICE;
            p.srcDevice = c.src;
        }
        if c.dst_host {
            p.dstMemoryType = driver::CU_MEMORYTYPE_HOST;
            p.dstHost = c.dst as usize as *mut c_void;
        } else {
            p.dstMemoryType = driver::CU_MEMORYTYPE_DEVICE;
            p.dstDevice = c.dst;
        }
        // SAFETY: the caller upholds the region contract; `p` is fully
        // initialised.
        check(unsafe { (api.cuMemcpy2DAsync)(&p, self.raw) }, "cuMemcpy2DAsync")
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if let (Ok(api), Ok(_g)) = (driver::api(), self.device.bind()) {
            // SAFETY: the stream was created by us and is destroyed once;
            // pending work completes before resources are released.
            unsafe { (api.cuStreamDestroy)(self.raw) };
        }
    }
}

/// A CUDA event (for cross-stream ordering and timing).
pub struct CudaEvent {
    raw: CUevent,
    device: Arc<CudaDevice>,
}

// SAFETY: event handles are thread-agnostic; calls bind the context.
unsafe impl Send for CudaEvent {}
// SAFETY: the driver API is thread-safe for events.
unsafe impl Sync for CudaEvent {}

impl std::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEvent").field("raw", &self.raw).finish()
    }
}

impl CudaEvent {
    /// Creates an event; `timing` enables [`CudaEvent::elapsed_ms`].
    pub fn new(device: &Arc<CudaDevice>, timing: bool) -> Result<Self> {
        let api = driver_api()?;
        let _g = device.bind()?;
        let flags = if timing { 0 } else { driver::CU_EVENT_DISABLE_TIMING };
        let mut raw: CUevent = std::ptr::null_mut();
        // SAFETY: context current, valid out pointer.
        check(unsafe { (api.cuEventCreate)(&mut raw, flags) }, "cuEventCreate")?;
        Ok(Self { raw, device: device.clone() })
    }

    /// Records the event on `stream`.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        if !Arc::ptr_eq(&self.device, stream.device()) {
            return Err(Error::cuda("event and stream belong to different devices"));
        }
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: valid handles on the same context.
        check(unsafe { (api.cuEventRecord)(self.raw, stream.raw()) }, "cuEventRecord")
    }

    /// Blocks until the event has completed.
    pub fn synchronize(&self) -> Result<()> {
        let api = driver_api()?;
        let _g = self.device.bind()?;
        // SAFETY: valid handle.
        check(unsafe { (api.cuEventSynchronize)(self.raw) }, "cuEventSynchronize")
    }

    /// Milliseconds between `start` and `self` (both recorded and complete).
    pub fn elapsed_ms(&self, start: &CudaEvent) -> Result<f32> {
        let api = driver_api()?;
        let _g = self.device.bind()?;
        let mut ms = 0f32;
        check(
            // SAFETY: valid handles, valid out pointer.
            unsafe { (api.cuEventElapsedTime)(&mut ms, start.raw, self.raw) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }

    /// Raw `CUevent`.
    pub fn raw(&self) -> CUevent {
        self.raw
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if let (Ok(api), Ok(_g)) = (driver::api(), self.device.bind()) {
            // SAFETY: created by us, destroyed once.
            unsafe { (api.cuEventDestroy)(self.raw) };
        }
    }
}

fn check_range(len: usize, offset: usize, n: usize) -> Result<()> {
    match offset.checked_add(n) {
        Some(end) if end <= len => Ok(()),
        _ => Err(Error::cuda(format!(
            "copy range {offset}..{offset}+{n} out of bounds for a {len}-byte buffer"
        ))),
    }
}
