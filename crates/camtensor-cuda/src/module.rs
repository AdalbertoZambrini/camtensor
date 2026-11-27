//! PTX modules and kernel launches.

use std::ffi::{CString, c_int, c_uint, c_void};
use std::sync::Arc;

use camtensor_core::{Error, Result};

use crate::device::CudaDevice;
use crate::error::{check, driver_api, error};
use crate::stream::CudaStream;
use crate::sys::driver::{self, CUfunction, CUmodule};

/// A loaded (JIT-compiled) CUDA module.
pub struct Module {
    raw: CUmodule,
    device: Arc<CudaDevice>,
}

// SAFETY: module handles are context-scoped, not thread-scoped.
unsafe impl Send for Module {}
// SAFETY: modules are immutable after loading.
unsafe impl Sync for Module {}

impl std::fmt::Debug for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Module").field("device", &self.device.ordinal()).finish()
    }
}

/// A kernel inside a [`Module`]. Valid while the module is alive.
#[derive(Debug, Clone, Copy)]
pub struct Function {
    raw: CUfunction,
}

// SAFETY: function handles are immutable, context-scoped values.
unsafe impl Send for Function {}
// SAFETY: as above.
unsafe impl Sync for Function {}

/// Launch geometry.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks).
    pub grid: (u32, u32, u32),
    /// Block dimensions (threads).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem: u32,
}

impl Module {
    /// JIT-compiles `ptx` for `device`.
    pub fn load_ptx(device: &Arc<CudaDevice>, ptx: &str) -> Result<Self> {
        let api = driver_api()?;
        let image = CString::new(ptx).map_err(|_| Error::cuda("PTX contains a NUL byte"))?;
        let _g = device.bind()?;
        let mut err_log = vec![0u8; 8192];
        let mut info_log = vec![0u8; 8192];
        let mut opts: [c_int; 4] = [
            driver::CU_JIT_ERROR_LOG_BUFFER,
            driver::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            driver::CU_JIT_INFO_LOG_BUFFER,
            driver::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut vals: [*mut c_void; 4] = [
            err_log.as_mut_ptr() as *mut c_void,
            err_log.len() as *mut c_void,
            info_log.as_mut_ptr() as *mut c_void,
            info_log.len() as *mut c_void,
        ];
        let mut raw: CUmodule = std::ptr::null_mut();
        // SAFETY: context current; `image` is a NUL-terminated PTX string;
        // option values follow the documented encoding (pointers and sizes
        // cast to `void*`), and the log buffers outlive the call.
        let r = unsafe {
            (api.cuModuleLoadDataEx)(
                &mut raw,
                image.as_ptr() as *const c_void,
                opts.len() as c_uint,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            )
        };
        if r != driver::CUDA_SUCCESS {
            let log = String::from_utf8_lossy(&err_log);
            let log = log.trim_end_matches('\0').trim();
            let e = error(r, "cuModuleLoadDataEx");
            return Err(if log.is_empty() { e } else { Error::cuda(format!("{e}: {log}")) });
        }
        Ok(Self { raw, device: device.clone() })
    }

    /// Looks up a kernel by its (unmangled, `extern "C"`) name.
    pub fn function(&self, name: &str) -> Result<Function> {
        let api = driver_api()?;
        let cname = CString::new(name).map_err(|_| Error::cuda("kernel name contains NUL"))?;
        let _g = self.device.bind()?;
        let mut raw: CUfunction = std::ptr::null_mut();
        check(
            // SAFETY: valid module, NUL-terminated name, valid out pointer.
            unsafe { (api.cuModuleGetFunction)(&mut raw, self.raw, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function { raw })
    }

    /// Owning device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Launches `func` on `stream`.
    ///
    /// # Safety
    /// `func` must belong to this module, and `params` must point to values
    /// whose types and order match the kernel's parameter list exactly. Any
    /// device memory the kernel touches must be valid for its execution.
    pub unsafe fn launch(
        &self,
        func: Function,
        cfg: LaunchConfig,
        stream: &CudaStream,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let api = driver_api()?;
        let _g = self.device.bind()?;
        check(
            // SAFETY: forwarded caller contract; `params` is a valid array of
            // parameter pointers for the duration of the call (the driver copies
            // parameter values at launch).
            unsafe {
                (api.cuLaunchKernel)(
                    func.raw,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shared_mem,
                    stream.raw(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let (Ok(api), Ok(_g)) = (driver::api(), self.device.bind()) {
            // SAFETY: loaded by us, unloaded once.
            unsafe { (api.cuModuleUnload)(self.raw) };
        }
    }
}
