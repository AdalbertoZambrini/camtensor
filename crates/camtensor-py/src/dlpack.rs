//! Zero-copy DLPack export (legacy `dltensor` and versioned
//! `dltensor_versioned` capsules, DLPack 1.0 ABI).

use std::ffi::{CStr, c_void};

use pyo3::ffi;
use pyo3::prelude::*;

use camtensor::{DType, Device, Tensor};

const NAME: &CStr = c"dltensor";
const NAME_VERSIONED: &CStr = c"dltensor_versioned";

const KDL_CPU: i32 = 1;
const KDL_CUDA: i32 = 2;
const KDL_UINT: u8 = 1;
const KDL_FLOAT: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

/// Owns the exported tensor (keeping its memory alive) and the shape arrays.
struct Ctx {
    _tensor: Tensor,
    shape: [i64; 4],
    strides: [i64; 4],
}

pub(crate) fn device(t: &Tensor) -> (i32, i32) {
    match t.device() {
        Device::Cuda(i) => (KDL_CUDA, i as i32),
        _ => (KDL_CPU, 0),
    }
}

fn dl_tensor(t: &Tensor, ctx: &mut Ctx) -> DLTensor {
    let (device_type, device_id) = device(t);
    let dtype = match t.dtype() {
        DType::U8 => DLDataType { code: KDL_UINT, bits: 8, lanes: 1 },
        DType::F16 => DLDataType { code: KDL_FLOAT, bits: 16, lanes: 1 },
        DType::F32 => DLDataType { code: KDL_FLOAT, bits: 32, lanes: 1 },
    };
    DLTensor {
        data: t.data_ptr() as *mut c_void,
        device: DLDevice { device_type, device_id },
        ndim: 4,
        dtype,
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn new_ctx(t: &Tensor) -> Box<Ctx> {
    let d = t.desc();
    Box::new(Ctx {
        _tensor: t.clone(),
        shape: d.shape.map(|v| v as i64),
        strides: d.strides.map(|v| v as i64),
    })
}

unsafe extern "C" fn delete_legacy(m: *mut DLManagedTensor) {
    if m.is_null() {
        return;
    }
    // SAFETY: `m` was created by `Box::into_raw` in `export` and the
    // DLPack protocol guarantees the deleter is called at most once.
    let m = unsafe { Box::from_raw(m) };
    // SAFETY: manager_ctx is the `Box<Ctx>` leaked in `export`.
    drop(unsafe { Box::from_raw(m.manager_ctx as *mut Ctx) });
}

unsafe extern "C" fn delete_versioned(m: *mut DLManagedTensorVersioned) {
    if m.is_null() {
        return;
    }
    // SAFETY: as in `delete_legacy`.
    let m = unsafe { Box::from_raw(m) };
    // SAFETY: manager_ctx is the `Box<Ctx>` leaked in `export`.
    drop(unsafe { Box::from_raw(m.manager_ctx as *mut Ctx) });
}

/// Capsule destructor: frees the tensor only if no consumer took ownership
/// (consumers rename the capsule to `used_dltensor[_versioned]`).
unsafe extern "C" fn capsule_destructor(cap: *mut ffi::PyObject) {
    // SAFETY: called by CPython with a valid capsule object.
    unsafe {
        if ffi::PyCapsule_IsValid(cap, NAME.as_ptr()) == 1 {
            let p = ffi::PyCapsule_GetPointer(cap, NAME.as_ptr()) as *mut DLManagedTensor;
            if let Some(d) = p.as_ref().and_then(|m| m.deleter) {
                d(p);
            }
        } else if ffi::PyCapsule_IsValid(cap, NAME_VERSIONED.as_ptr()) == 1 {
            let p = ffi::PyCapsule_GetPointer(cap, NAME_VERSIONED.as_ptr())
                as *mut DLManagedTensorVersioned;
            if let Some(d) = p.as_ref().and_then(|m| m.deleter) {
                d(p);
            }
        } else {
            // Consumed: the consumer owns the tensor now. Clear any error
            // raised by the failed IsValid checks.
            ffi::PyErr_Clear();
        }
    }
}

/// Exports `t` as a DLPack capsule.
pub(crate) fn export<'py>(
    py: Python<'py>,
    t: &Tensor,
    versioned: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let mut ctx = new_ctx(t);
    let dl = dl_tensor(t, &mut ctx);
    let ctx_ptr = Box::into_raw(ctx) as *mut c_void;
    let (ptr, name) = if versioned {
        let m = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: ctx_ptr,
            deleter: Some(delete_versioned),
            flags: 0,
            dl_tensor: dl,
        });
        (Box::into_raw(m) as *mut c_void, NAME_VERSIONED)
    } else {
        let m = Box::new(DLManagedTensor {
            dl_tensor: dl,
            manager_ctx: ctx_ptr,
            deleter: Some(delete_legacy),
        });
        (Box::into_raw(m) as *mut c_void, NAME)
    };
    // SAFETY: `ptr` is a valid heap pointer, `name` is a static C string and
    // the destructor matches the PyCapsule_Destructor signature.
    let cap = unsafe { ffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_destructor)) };
    if cap.is_null() {
        // SAFETY: capsule creation failed, so we still own `ptr`.
        unsafe {
            if versioned {
                delete_versioned(ptr as *mut DLManagedTensorVersioned);
            } else {
                delete_legacy(ptr as *mut DLManagedTensor);
            }
        }
        return Err(PyErr::fetch(py));
    }
    // SAFETY: `cap` is a new (owned) reference.
    Ok(unsafe { Bound::from_owned_ptr(py, cap) })
}
