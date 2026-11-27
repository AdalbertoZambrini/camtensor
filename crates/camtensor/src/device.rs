//! Device glue: allocations, uploads and the preprocessing dispatch between
//! the CPU implementation and the CUDA kernels.

use std::sync::Arc;

use camtensor_core::{Affine2, Device, Error, Frame, FrameData, Result, TensorDesc, TensorSpec};

use crate::output::{AllocFn, Slot, ZeroFn};
use crate::tensor::{AlignedBuf, Tensor, TensorStorage};

#[cfg(feature = "cuda")]
pub(crate) mod gpu;

/// Resolved compute backend of an engine.
#[derive(Clone)]
pub(crate) enum Backend {
    /// CPU only.
    Cpu,
    /// CUDA device available.
    #[cfg(feature = "cuda")]
    Cuda(Arc<gpu::Gpu>),
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Backend({})", self.device())
    }
}

impl Backend {
    /// Resolves `Device::Auto` and opens the CUDA device if requested.
    pub(crate) fn resolve(device: Device) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Backend::Cpu),
            Device::Auto => {
                #[cfg(feature = "cuda")]
                if let Ok(g) = gpu::Gpu::open(0) {
                    return Ok(Backend::Cuda(Arc::new(g)));
                }
                Ok(Backend::Cpu)
            }
            Device::Cuda(i) => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Backend::Cuda(Arc::new(gpu::Gpu::open(i)?)))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::unavailable(format!(
                        "cuda:{i} requested but camtensor was built without the `cuda` feature"
                    )))
                }
            }
        }
    }

    /// Device used for decoding and default outputs.
    pub(crate) fn device(&self) -> Device {
        match self {
            Backend::Cpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            Backend::Cuda(g) => Device::Cuda(g.ordinal()),
        }
    }

    /// Checks that an output may live on `device` and normalises it.
    pub(crate) fn output_device(&self, requested: Option<Device>) -> Result<Device> {
        match requested.unwrap_or(Device::Auto) {
            Device::Auto => Ok(self.device()),
            Device::Cpu => Ok(Device::Cpu),
            Device::Cuda(i) => {
                if self.device() == Device::Cuda(i) {
                    Ok(Device::Cuda(i))
                } else {
                    Err(Error::config(format!(
                        "output device cuda:{i} differs from the engine device {}",
                        self.device()
                    )))
                }
            }
        }
    }

    /// Allocation function for tensors on `device`.
    pub(crate) fn alloc_fn(&self, device: Device) -> Box<AllocFn> {
        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let Backend::Cuda(g) = self else { unreachable!("validated by output_device") };
                let g = g.clone();
                Box::new(move |n| g.alloc_tensor(n))
            }
            _ => Box::new(|n| Ok(TensorStorage::Host(AlignedBuf::zeroed(n)))),
        }
    }

    /// Zero-fill function for tensors on `device`.
    pub(crate) fn zero_fn(&self, device: Device) -> Box<ZeroFn> {
        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let Backend::Cuda(g) = self else { unreachable!("validated by output_device") };
                let g = g.clone();
                Box::new(move |addr, len| g.memset_zero(addr, len))
            }
            _ => Box::new(|addr, len| {
                // SAFETY: `addr` is the start of a host slot of at least `len`
                // bytes inside a live batch buffer owned by the caller.
                unsafe { std::ptr::write_bytes(addr as *mut u8, 0, len) };
                Ok(())
            }),
        }
    }

    /// Per-thread execution context (CUDA stream, staging buffers).
    pub(crate) fn worker(&self) -> Result<WorkerCtx> {
        Ok(WorkerCtx {
            #[cfg(feature = "cuda")]
            gpu: match self {
                Backend::Cuda(g) => Some(g.worker()?),
                Backend::Cpu => None,
            },
        })
    }
}

/// Per-thread context used by stream workers and crop calls.
pub(crate) struct WorkerCtx {
    #[cfg(feature = "cuda")]
    gpu: Option<gpu::Worker>,
}

impl WorkerCtx {
    /// Makes `frame` usable on `device` (uploads host frames to the GPU
    /// once so that several outputs and later crops share the copy).
    pub(crate) fn frame_for(&mut self, frame: &Frame, device: Device) -> Result<Frame> {
        match (&frame.data, device) {
            (FrameData::Host(_), Device::Cuda(_)) => {
                #[cfg(feature = "cuda")]
                if let Some(g) = &mut self.gpu {
                    return g.upload_frame(frame);
                }
                Err(Error::unavailable("cuda output requested without a CUDA context"))
            }
            _ => Ok(frame.clone()),
        }
    }

    /// Preprocesses `frame` into the destination `slot` (on `out_device`).
    pub(crate) fn write_image(
        &mut self,
        frame: &Frame,
        sampling: &Affine2,
        spec: &TensorSpec,
        slot: Slot,
        out_device: Device,
    ) -> Result<()> {
        match (&frame.data, out_device) {
            (FrameData::Host(bytes), Device::Cpu | Device::Auto) => {
                // SAFETY: the batcher reserved `slot` exclusively for this
                // writer; it lies inside a live host buffer of `slot.len` bytes.
                let out = unsafe { std::slice::from_raw_parts_mut(slot.addr as *mut u8, slot.len) };
                camtensor_preprocess::preprocess_into(&frame.desc, bytes, sampling, spec, out)
            }
            #[cfg(feature = "cuda")]
            (_, _) => match &mut self.gpu {
                Some(g) => g.write_image(frame, &[(*sampling, slot.addr)], spec, out_device),
                None => Err(Error::unavailable("device frame without a CUDA context")),
            },
            #[cfg(not(feature = "cuda"))]
            (_, _) => Err(Error::unavailable("device frames require the `cuda` feature")),
        }
    }

    /// Copies device memory at `src` into `dst`.
    pub(crate) fn download(&mut self, src: u64, dst: &mut [u8]) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(g) = &mut self.gpu {
            return g.download(src, dst);
        }
        let _ = (src, dst);
        Err(Error::unavailable("device tensor without a CUDA context"))
    }

    /// Extracts crops of `frame` into a new tensor `[rois.len(), ...]`.
    pub(crate) fn crop(
        &mut self,
        backend: &Backend,
        frame: &Frame,
        plans: &[Affine2],
        spec: &TensorSpec,
        out_device: Device,
    ) -> Result<Tensor> {
        let n = plans.len();
        let desc = TensorDesc::contiguous(out_device, spec.dtype, spec.batch_shape(n));
        let storage = Arc::new((backend.alloc_fn(out_device))(desc.byte_len().max(1))?);
        let img = spec.image_bytes();
        match (&frame.data, out_device) {
            (FrameData::Host(bytes), Device::Cpu | Device::Auto) => {
                for (i, sampling) in plans.iter().enumerate() {
                    let addr = storage.addr() as usize + i * img;
                    // SAFETY: fresh allocation of n * img bytes owned by `storage`.
                    let out = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, img) };
                    camtensor_preprocess::preprocess_into(&frame.desc, bytes, sampling, spec, out)?;
                }
            }
            #[cfg(feature = "cuda")]
            _ => {
                let g = self.gpu.as_mut().ok_or_else(|| Error::unavailable("no CUDA context"))?;
                let items: Vec<(Affine2, u64)> = plans
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (*s, storage.addr() + (i * img) as u64))
                    .collect();
                g.write_image(frame, &items, spec, out_device)?;
            }
            #[cfg(not(feature = "cuda"))]
            _ => return Err(Error::unavailable("device frames require the `cuda` feature")),
        }
        Ok(Tensor::new(desc, storage, 0))
    }
}
