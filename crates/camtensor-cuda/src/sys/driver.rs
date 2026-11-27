//! CUDA driver API (`nvcuda.dll` / `libcuda.so.1`), subset.

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use super::{dynamic_api, open_library};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;

#[repr(C)]
pub struct CUctx_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUstream_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUevent_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUmod_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUfunc_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUarray_st {
    _p: [u8; 0],
}

pub type CUcontext = *mut CUctx_st;
pub type CUstream = *mut CUstream_st;
pub type CUevent = *mut CUevent_st;
pub type CUmodule = *mut CUmod_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUarray = *mut CUarray_st;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_INSUFFICIENT_DRIVER: CUresult = 35;
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 1;

pub const CU_MEMORYTYPE_HOST: c_uint = 1;
pub const CU_MEMORYTYPE_DEVICE: c_uint = 2;

pub const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;

/// `CUDA_MEMCPY2D` (identical on LLP64 and LP64: only `size_t` and pointers).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: c_uint,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: c_uint,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}

impl Default for CUDA_MEMCPY2D {
    fn default() -> Self {
        Self {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: 0,
            srcHost: std::ptr::null(),
            srcDevice: 0,
            srcArray: std::ptr::null_mut(),
            srcPitch: 0,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: 0,
            dstHost: std::ptr::null_mut(),
            dstDevice: 0,
            dstArray: std::ptr::null_mut(),
            dstPitch: 0,
            WidthInBytes: 0,
            Height: 0,
        }
    }
}

dynamic_api! {
    /// Function table of the CUDA driver API.
    pub struct DriverApi {
        cuInit = "cuInit": fn(c_uint) -> CUresult;
        cuDriverGetVersion = "cuDriverGetVersion": fn(*mut c_int) -> CUresult;
        cuGetErrorName = "cuGetErrorName": fn(CUresult, *mut *const c_char) -> CUresult;
        cuGetErrorString = "cuGetErrorString": fn(CUresult, *mut *const c_char) -> CUresult;
        cuDeviceGetCount = "cuDeviceGetCount": fn(*mut c_int) -> CUresult;
        cuDeviceGet = "cuDeviceGet": fn(*mut CUdevice, c_int) -> CUresult;
        cuDeviceGetName = "cuDeviceGetName": fn(*mut c_char, c_int, CUdevice) -> CUresult;
        cuDeviceGetAttribute = "cuDeviceGetAttribute": fn(*mut c_int, c_int, CUdevice) -> CUresult;
        cuDeviceTotalMem = "cuDeviceTotalMem_v2": fn(*mut usize, CUdevice) -> CUresult;
        cuDevicePrimaryCtxRetain = "cuDevicePrimaryCtxRetain": fn(*mut CUcontext, CUdevice) -> CUresult;
        cuDevicePrimaryCtxRelease = "cuDevicePrimaryCtxRelease_v2": fn(CUdevice) -> CUresult;
        cuCtxPushCurrent = "cuCtxPushCurrent_v2": fn(CUcontext) -> CUresult;
        cuCtxPopCurrent = "cuCtxPopCurrent_v2": fn(*mut CUcontext) -> CUresult;
        cuCtxSynchronize = "cuCtxSynchronize": fn() -> CUresult;
        cuMemGetInfo = "cuMemGetInfo_v2": fn(*mut usize, *mut usize) -> CUresult;
        cuStreamCreate = "cuStreamCreate": fn(*mut CUstream, c_uint) -> CUresult;
        cuStreamDestroy = "cuStreamDestroy_v2": fn(CUstream) -> CUresult;
        cuStreamSynchronize = "cuStreamSynchronize": fn(CUstream) -> CUresult;
        cuStreamQuery = "cuStreamQuery": fn(CUstream) -> CUresult;
        cuStreamWaitEvent = "cuStreamWaitEvent": fn(CUstream, CUevent, c_uint) -> CUresult;
        cuEventCreate = "cuEventCreate": fn(*mut CUevent, c_uint) -> CUresult;
        cuEventDestroy = "cuEventDestroy_v2": fn(CUevent) -> CUresult;
        cuEventRecord = "cuEventRecord": fn(CUevent, CUstream) -> CUresult;
        cuEventSynchronize = "cuEventSynchronize": fn(CUevent) -> CUresult;
        cuEventQuery = "cuEventQuery": fn(CUevent) -> CUresult;
        cuEventElapsedTime = "cuEventElapsedTime": fn(*mut f32, CUevent, CUevent) -> CUresult;
        cuMemAlloc = "cuMemAlloc_v2": fn(*mut CUdeviceptr, usize) -> CUresult;
        cuMemFree = "cuMemFree_v2": fn(CUdeviceptr) -> CUresult;
        cuMemHostAlloc = "cuMemHostAlloc": fn(*mut *mut c_void, usize, c_uint) -> CUresult;
        cuMemFreeHost = "cuMemFreeHost": fn(*mut c_void) -> CUresult;
        cuMemcpyHtoD = "cuMemcpyHtoD_v2": fn(CUdeviceptr, *const c_void, usize) -> CUresult;
        cuMemcpyDtoH = "cuMemcpyDtoH_v2": fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
        cuMemcpyHtoDAsync = "cuMemcpyHtoDAsync_v2": fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
        cuMemcpyDtoHAsync = "cuMemcpyDtoHAsync_v2": fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
        cuMemcpyDtoDAsync = "cuMemcpyDtoDAsync_v2": fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
        cuMemcpy2DAsync = "cuMemcpy2DAsync_v2": fn(*const CUDA_MEMCPY2D, CUstream) -> CUresult;
        cuMemsetD8Async = "cuMemsetD8Async": fn(CUdeviceptr, u8, usize, CUstream) -> CUresult;
        cuModuleLoadDataEx = "cuModuleLoadDataEx": fn(*mut CUmodule, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> CUresult;
        cuModuleUnload = "cuModuleUnload": fn(CUmodule) -> CUresult;
        cuModuleGetFunction = "cuModuleGetFunction": fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
        cuLaunchKernel = "cuLaunchKernel": fn(
            CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream,
            *mut *mut c_void, *mut *mut c_void,
        ) -> CUresult;
    }
}

#[cfg(windows)]
const DRIVER_LIBS: &[&str] = &["nvcuda.dll"];
#[cfg(not(windows))]
const DRIVER_LIBS: &[&str] = &["libcuda.so.1", "libcuda.so"];

static DRIVER: OnceLock<Result<DriverApi, String>> = OnceLock::new();

/// Loads the driver library and calls `cuInit(0)` once per process.
pub fn api() -> Result<&'static DriverApi, &'static str> {
    DRIVER
        .get_or_init(|| {
            let lib = open_library(DRIVER_LIBS)?;
            // SAFETY: `lib` is the CUDA driver; the declared signatures match
            // the CUDA driver API (cuda.h).
            let api = unsafe { DriverApi::load(lib)? };
            // SAFETY: cuInit has no preconditions.
            let r = unsafe { (api.cuInit)(0) };
            if r != CUDA_SUCCESS {
                return Err(format!("cuInit failed: {}", error_name(&api, r)));
            }
            Ok(api)
        })
        .as_ref()
        .map_err(|e| e.as_str())
}

/// Readable name of a `CUresult` (e.g. `CUDA_ERROR_OUT_OF_MEMORY`).
pub fn error_name(api: &DriverApi, r: CUresult) -> String {
    let mut name: *const c_char = std::ptr::null();
    let mut desc: *const c_char = std::ptr::null();
    // SAFETY: both calls only write a pointer to a static string (or leave
    // it untouched for unknown codes).
    unsafe {
        (api.cuGetErrorName)(r, &mut name);
        (api.cuGetErrorString)(r, &mut desc);
    }
    let to_string = |p: *const c_char| {
        if p.is_null() {
            None
        } else {
            // SAFETY: the driver returns NUL-terminated static strings.
            Some(unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned())
        }
    };
    match (to_string(name), to_string(desc)) {
        (Some(n), Some(d)) => format!("{n} ({r}): {d}"),
        (Some(n), None) => format!("{n} ({r})"),
        _ => format!("CUresult {r}"),
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn memcpy2d_layout() {
        assert_eq!(size_of::<CUDA_MEMCPY2D>(), 128);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, srcMemoryType), 16);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, srcHost), 24);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, srcDevice), 32);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, srcPitch), 48);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, dstXInBytes), 56);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, dstMemoryType), 72);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, dstDevice), 88);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, dstPitch), 104);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, WidthInBytes), 112);
        assert_eq!(offset_of!(CUDA_MEMCPY2D, Height), 120);
    }
}
