//! CUDA glue: device tensors, uploads and the fused preprocessing kernel.

use std::sync::Arc;

use camtensor_core::{Affine2, Device, Error, Frame, FrameData, Result, TensorSpec};
use camtensor_cuda::{
    BufferPool, CudaDevice, CudaStream, DeviceBuffer, Preprocessor, SampleItem, SourceImage,
};

use crate::tensor::TensorStorage;

/// An opened CUDA device with its preprocessing kernel loaded.
pub(crate) struct Gpu {
    device: Arc<CudaDevice>,
    pre: Arc<Preprocessor>,
}

impl Gpu {
    pub(crate) fn open(ordinal: u32) -> Result<Self> {
        if !camtensor_cuda::is_available() {
            return Err(Error::unavailable("CUDA driver not found"));
        }
        let device = CudaDevice::get(ordinal)?;
        let pre = Preprocessor::get(&device)?;
        tracing::info!(gpu = device.name(), ordinal, "CUDA device ready");
        Ok(Self { device, pre })
    }

    pub(crate) fn ordinal(&self) -> u32 {
        self.device.ordinal()
    }

    pub(crate) fn alloc_tensor(&self, len: usize) -> Result<TensorStorage> {
        Ok(TensorStorage::Device(Box::new(DeviceBuffer::zeroed(&self.device, len.max(1))?)))
    }

    pub(crate) fn memset_zero(&self, addr: u64, len: usize) -> Result<()> {
        let stream = CudaStream::new(&self.device)?;
        let zeros = vec![0u8; len];
        // SAFETY: `addr..addr+len` is a slot inside a live device batch
        // buffer; `zeros` outlives the copy because we synchronise.
        unsafe { stream.memcpy_htod_async(addr, zeros.as_ptr(), len)? };
        stream.synchronize()
    }

    pub(crate) fn worker(&self) -> Result<Worker> {
        Ok(Worker {
            device: self.device.clone(),
            pre: self.pre.clone(),
            stream: CudaStream::new(&self.device)?,
            upload_pool: None,
            scratch: None,
        })
    }
}

/// Per-thread GPU context: one CUDA stream plus staging buffers.
pub(crate) struct Worker {
    device: Arc<CudaDevice>,
    pre: Arc<Preprocessor>,
    stream: CudaStream,
    upload_pool: Option<BufferPool>,
    scratch: Option<DeviceBuffer>,
}

impl Worker {
    /// Uploads a host frame to a pooled device buffer.
    pub(crate) fn upload_frame(&mut self, frame: &Frame) -> Result<Frame> {
        let FrameData::Host(bytes) = &frame.data else { return Ok(frame.clone()) };
        let need = frame.desc.byte_len().min(bytes.len());
        if self.upload_pool.as_ref().is_none_or(|p| p.byte_len() < need) {
            self.upload_pool = Some(BufferPool::new(&self.device, need, 8));
        }
        let buf = self.upload_pool.as_ref().expect("pool").get()?;
        self.stream.upload(&buf, 0, &bytes[..need])?;
        Ok(Frame { desc: frame.desc, data: FrameData::Device(buf), pts: frame.pts })
    }

    /// Preprocesses `frame` (device memory) into each `(sampling, addr)`
    /// destination. Destinations are device pointers when `out_device` is
    /// CUDA and host pointers otherwise (rendered on the GPU, then copied).
    pub(crate) fn write_image(
        &mut self,
        frame: &Frame,
        items: &[(Affine2, u64)],
        spec: &TensorSpec,
        out_device: Device,
    ) -> Result<()> {
        let frame = match &frame.data {
            FrameData::Host(_) => self.upload_frame(frame)?,
            FrameData::Device(m) if m.device_ordinal() != self.device.ordinal() => {
                return Err(Error::cuda("frame lives on another CUDA device"));
            }
            FrameData::Device(_) => frame.clone(),
        };
        let src = SourceImage::from_frame(&frame)?;
        let img = spec.image_bytes();
        match out_device {
            Device::Cuda(_) => {
                let samples: Vec<SampleItem> = items
                    .iter()
                    .map(|(s, addr)| SampleItem { sampling: s.0, dst_ptr: *addr })
                    .collect();
                // SAFETY: each destination is a distinct, reserved slot of
                // `img` bytes inside a live device buffer on this device; the
                // source frame is kept alive by `frame` until we synchronise.
                unsafe { self.pre.preprocess(&self.stream, &src, &samples, spec)? };
                self.stream.synchronize()
            }
            _ => {
                let need = img * items.len();
                if self.scratch.as_ref().is_none_or(|b| b.byte_len() < need) {
                    self.scratch = Some(DeviceBuffer::alloc(&self.device, need.max(1))?);
                }
                let scratch = self.scratch.as_ref().expect("scratch");
                let samplings: Vec<Affine2> = items.iter().map(|(s, _)| *s).collect();
                self.pre.preprocess_into(&self.stream, &src, &samplings, scratch, 0, spec)?;
                for (i, (_, addr)) in items.iter().enumerate() {
                    // SAFETY: `addr` is a reserved host slot of `img` bytes;
                    // the scratch range is inside the allocation; we
                    // synchronise before the slot is published.
                    unsafe {
                        self.stream.memcpy_dtoh_async(
                            *addr as *mut u8,
                            scratch.device_ptr() + (i * img) as u64,
                            img,
                        )?
                    };
                }
                self.stream.synchronize()
            }
        }
    }

    /// Copies device memory at `src` into `dst`.
    pub(crate) fn download(&mut self, src: u64, dst: &mut [u8]) -> Result<()> {
        // SAFETY: `src` addresses a live device tensor of at least
        // `dst.len()` bytes (checked by the caller); we synchronise before
        // returning so `dst` outlives the copy.
        unsafe { self.stream.memcpy_dtoh_async(dst.as_mut_ptr(), src, dst.len())? };
        self.stream.synchronize()
    }
}
