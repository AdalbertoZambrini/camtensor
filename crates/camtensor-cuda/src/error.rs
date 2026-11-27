//! Mapping of `CUresult` codes to [`camtensor_core::Error`].

use camtensor_core::{Error, Result};

use crate::sys::driver::{self, CUresult};

/// Converts a `CUresult` into a `Result`, naming the failed call.
pub(crate) fn check(r: CUresult, what: &str) -> Result<()> {
    if r == driver::CUDA_SUCCESS {
        return Ok(());
    }
    Err(error(r, what))
}

/// Builds the error for a failed call.
pub(crate) fn error(r: CUresult, what: &str) -> Error {
    let name = match driver::api() {
        Ok(api) => driver::error_name(api, r),
        Err(_) => format!("CUresult {r}"),
    };
    let msg = format!("{what} failed: {name}");
    match r {
        driver::CUDA_ERROR_NO_DEVICE
        | driver::CUDA_ERROR_INSUFFICIENT_DRIVER
        | driver::CUDA_ERROR_NOT_SUPPORTED => Error::Unavailable(msg),
        _ => Error::Cuda(msg),
    }
}

/// Driver API or an `Unavailable` error.
pub(crate) fn driver_api() -> Result<&'static driver::DriverApi> {
    driver::api().map_err(|e| Error::unavailable(format!("CUDA driver not available: {e}")))
}
