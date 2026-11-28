//! Multi-camera batcher: collects preprocessed images from many streams into
//! pooled batch buffers and delivers them to consumers.
//!
//! Writers (one per stream) reserve a slot under a short lock, write the
//! image *outside* the lock directly into the batch buffer (no extra copy),
//! then commit. A batch is delivered when it is full or when `max_wait_ms`
//! expired since its first image. Buffers are recycled once every holder
//! (consumer batches, exported tensors) has dropped them.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use serde::Serialize;

use camtensor_core::{
    Backpressure, Device, Error, FrameMeta, OutputConfig, Result, TensorDesc, TensorSpec,
};

use crate::tensor::{Batch, BatchItem, Tensor, TensorStorage};

/// Allocates a zeroed buffer of the given size on the output's device.
pub(crate) type AllocFn = dyn Fn(usize) -> Result<TensorStorage> + Send + Sync;
/// Zero-fills `len` bytes at `addr` on the output's device.
pub(crate) type ZeroFn = dyn Fn(u64, usize) -> Result<()> + Send + Sync;

/// Completion of asynchronous device work for one image: waiting on it
/// guarantees the image is fully written; dropping it releases what the work
/// still needed (e.g. the source frame).
pub(crate) type Fence = Box<dyn FnOnce() + Send>;

/// What a writer returns for a successfully written image.
pub(crate) type Written = (FrameMeta, Option<camtensor_core::Frame>, Option<Fence>);

/// A finished batch waiting for its consumer, with the GPU work it depends on.
struct Ready {
    batch: Batch,
    fences: Vec<Fence>,
}

impl Ready {
    /// Waits for all device work of the batch, then returns it.
    fn complete(self) -> Batch {
        for f in self.fences {
            f();
        }
        self.batch
    }
}

/// Where a writer must put its image.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Slot {
    /// Address of the image (host pointer or device pointer).
    pub addr: u64,
    /// Size of the image in bytes.
    pub len: usize,
}

/// Counters of an output.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OutputStats {
    /// Batches delivered to the queue.
    pub batches: u64,
    /// Valid images delivered.
    pub images: u64,
    /// Images dropped because of backpressure.
    pub dropped_images: u64,
    /// Slots that failed to be written (preprocess errors).
    pub failed_images: u64,
    /// Buffers allocated so far.
    pub buffers: usize,
    /// Batches currently queued and not yet received.
    pub queued: usize,
}

struct Filling {
    storage: Arc<TensorStorage>,
    items: Vec<Option<BatchItem>>,
    fences: Vec<Fence>,
    reserved: usize,
    done: usize,
    sealed: bool,
    first: Instant,
    id: u64,
}

struct State {
    pool: Vec<Arc<TensorStorage>>,
    filling: Option<Filling>,
    sealing: Vec<Filling>,
    ready: VecDeque<Ready>,
    pending_drops: HashMap<Arc<str>, u64>,
    closed: bool,
    next_id: u64,
    stats: OutputStats,
}

pub(crate) struct Output {
    pub(crate) name: Arc<str>,
    pub(crate) cfg: OutputConfig,
    pub(crate) device: Device,
    alloc: Box<AllocFn>,
    zero: Box<ZeroFn>,
    state: Mutex<State>,
    cv_ready: Condvar,
    cv_space: Condvar,
    cv_flush: Condvar,
    batch_seq: AtomicU64,
}

impl std::fmt::Debug for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Output").field("name", &self.name).field("device", &self.device).finish()
    }
}

impl Output {
    pub(crate) fn new(
        name: &str,
        cfg: OutputConfig,
        device: Device,
        alloc: Box<AllocFn>,
        zero: Box<ZeroFn>,
    ) -> Result<Arc<Self>> {
        cfg.validate()?;
        let out = Arc::new(Self {
            name: name.into(),
            cfg,
            device,
            alloc,
            zero,
            state: Mutex::new(State {
                pool: Vec::new(),
                filling: None,
                sealing: Vec::new(),
                ready: VecDeque::new(),
                pending_drops: HashMap::new(),
                closed: false,
                next_id: 0,
                stats: OutputStats::default(),
            }),
            cv_ready: Condvar::new(),
            cv_space: Condvar::new(),
            cv_flush: Condvar::new(),
            batch_seq: AtomicU64::new(0),
        });
        if out.cfg.batch.max_size > 1 && out.cfg.batch.max_wait_ms > 0 {
            let weak = Arc::downgrade(&out);
            std::thread::Builder::new()
                .name(format!("ct-flush-{name}"))
                .spawn(move || flusher(weak))
                .map_err(Error::Io)?;
        }
        Ok(out)
    }

    pub(crate) fn spec(&self) -> &TensorSpec {
        &self.cfg.tensor
    }

    fn image_bytes(&self) -> usize {
        self.cfg.tensor.image_bytes()
    }

    fn buffer_bytes(&self) -> usize {
        self.image_bytes() * self.cfg.batch.max_size
    }

    /// Maximum number of buffers: queue + one being filled + one sealing.
    fn max_buffers(&self) -> usize {
        self.cfg.queue_depth + 2
    }

    /// Records a frame of `stream` that was skipped for this output before
    /// reaching the batcher (e.g. `max_fps`), so that the next delivered
    /// image reports it in `dropped_since_last`.
    pub(crate) fn note_drop(&self, stream: &Arc<str>, n: u64) {
        let mut st = self.state.lock();
        *st.pending_drops.entry(stream.clone()).or_default() += n;
    }

    /// Writes one image. `write` receives the destination slot and returns
    /// the image metadata (or an error, which leaves the slot invalid).
    ///
    /// Returns `Ok(false)` when the image was dropped by backpressure.
    pub(crate) fn submit(
        &self,
        stream: &Arc<str>,
        write: impl FnOnce(Slot) -> Result<Written>,
    ) -> Result<bool> {
        let (id, idx, storage) = {
            let mut st = self.state.lock();
            loop {
                if st.closed {
                    return Err(Error::Closed);
                }
                if st.filling.is_some() {
                    break;
                }
                match self.acquire_buffer(&mut st)? {
                    Some(storage) => {
                        let id = st.next_id;
                        st.next_id += 1;
                        st.filling = Some(Filling {
                            storage,
                            items: vec![None; self.cfg.batch.max_size],
                            fences: Vec::new(),
                            reserved: 0,
                            done: 0,
                            sealed: false,
                            first: Instant::now(),
                            id,
                        });
                        break;
                    }
                    None => match self.cfg.backpressure {
                        Backpressure::Block => {
                            self.cv_space.wait_for(&mut st, Duration::from_millis(100));
                        }
                        Backpressure::DropOldest | Backpressure::LatestOnly => {
                            if !drop_oldest(&mut st) {
                                // Every buffer is held by the consumer: drop this image.
                                st.stats.dropped_images += 1;
                                *st.pending_drops.entry(stream.clone()).or_default() += 1;
                                return Ok(false);
                            }
                        }
                    },
                }
            }
            let max = self.cfg.batch.max_size;
            let f = st.filling.as_mut().expect("filling");
            let idx = f.reserved;
            f.reserved += 1;
            if f.reserved == 1 {
                f.first = Instant::now();
            }
            let res = (f.id, idx, f.storage.clone());
            if f.reserved == max {
                f.sealed = true;
                let f = st.filling.take().expect("filling");
                st.sealing.push(f);
            } else if f.reserved == 1 {
                self.cv_flush.notify_one();
            }
            res
        };

        let slot = Slot {
            addr: storage.addr() + (idx * self.image_bytes()) as u64,
            len: self.image_bytes(),
        };
        let result = write(slot);
        drop(storage);

        let mut st = self.state.lock();
        let mut fence = None;
        let item = match result {
            Ok((mut meta, frame, f)) => {
                if let Some(n) = st.pending_drops.remove(stream) {
                    meta.dropped_since_last += n;
                }
                fence = f;
                Some(BatchItem { meta, valid: true, frame })
            }
            Err(e) => {
                st.stats.failed_images += 1;
                tracing::warn!(output = %self.name, stream = %stream, "preprocess failed: {e}");
                None
            }
        };
        let finished = {
            let f = find_filling(&mut st, id).expect("slot owner");
            f.items[idx] = item;
            f.fences.extend(fence);
            f.done += 1;
            f.sealed && f.done == f.reserved
        };
        if finished {
            let f = take_filling(&mut st, id).expect("slot owner");
            self.finalize(&mut st, f);
        }
        Ok(true)
    }

    fn acquire_buffer(&self, st: &mut State) -> Result<Option<Arc<TensorStorage>>> {
        if let Some(b) = st.pool.iter().find(|b| Arc::strong_count(b) == 1) {
            return Ok(Some(b.clone()));
        }
        if st.pool.len() < self.max_buffers() {
            let b = Arc::new((self.alloc)(self.buffer_bytes())?);
            st.pool.push(b.clone());
            st.stats.buffers = st.pool.len();
            return Ok(Some(b));
        }
        Ok(None)
    }

    fn finalize(&self, st: &mut State, f: Filling) {
        let spec = &self.cfg.tensor;
        let n = if self.cfg.batch.pad_to_max { self.cfg.batch.max_size } else { f.reserved };
        let img = self.image_bytes();
        let mut items = Vec::with_capacity(n);
        let mut valid = 0u64;
        for (i, it) in f.items.into_iter().take(n).enumerate() {
            match it {
                Some(item) => {
                    valid += 1;
                    items.push(item);
                }
                None => {
                    // Padding or failed slot: zero it so consumers see clean data.
                    let addr = f.storage.addr() + (i * img) as u64;
                    if let Err(e) = (self.zero)(addr, img) {
                        tracing::warn!("zero-fill failed: {e}");
                    }
                    items.push(BatchItem { meta: placeholder_meta(), valid: false, frame: None });
                }
            }
        }
        if valid == 0 {
            for fence in f.fences {
                fence();
            }
            return;
        }
        let desc = TensorDesc::contiguous(self.device, spec.dtype, spec.batch_shape(n));
        let batch = Batch {
            output: self.name.clone(),
            tensor: Tensor::new(desc, f.storage, 0),
            items,
            seq: self.batch_seq.fetch_add(1, Ordering::Relaxed),
        };
        st.stats.batches += 1;
        st.stats.images += valid;
        st.ready.push_back(Ready { batch, fences: f.fences });
        let limit = match self.cfg.backpressure {
            Backpressure::LatestOnly => 1,
            _ => self.cfg.queue_depth,
        };
        while st.ready.len() > limit {
            if self.cfg.backpressure == Backpressure::Block {
                break;
            }
            drop_oldest(st);
        }
        st.stats.queued = st.ready.len();
        self.cv_ready.notify_all();
    }

    /// Seals the batch being filled if its deadline expired. Returns the
    /// time until the next deadline.
    fn flush_due(&self, st: &mut State) -> Option<Duration> {
        let wait = Duration::from_millis(self.cfg.batch.max_wait_ms);
        let f = st.filling.as_ref()?;
        if f.reserved == 0 {
            return None;
        }
        let elapsed = f.first.elapsed();
        if elapsed < wait {
            return Some(wait - elapsed);
        }
        let mut f = st.filling.take().expect("filling");
        f.sealed = true;
        if f.done == f.reserved {
            self.finalize(st, f);
        } else {
            st.sealing.push(f);
        }
        None
    }

    /// Blocks until a batch is available, the output is closed, or the
    /// timeout expires.
    pub(crate) fn recv(&self, timeout: Option<Duration>) -> Result<Batch> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut st = self.state.lock();
        loop {
            if let Some(r) = st.ready.pop_front() {
                st.stats.queued = st.ready.len();
                self.cv_space.notify_all();
                drop(st);
                // Wait for the GPU outside the lock.
                return Ok(r.complete());
            }
            if st.closed {
                return Err(Error::Closed);
            }
            match deadline {
                Some(d) => {
                    if self.cv_ready.wait_until(&mut st, d).timed_out() && st.ready.is_empty() {
                        return Err(Error::Timeout);
                    }
                }
                None => self.cv_ready.wait(&mut st),
            }
        }
    }

    /// Non-blocking receive.
    pub(crate) fn try_recv(&self) -> Option<Batch> {
        let mut st = self.state.lock();
        let r = st.ready.pop_front();
        if r.is_some() {
            st.stats.queued = st.ready.len();
            self.cv_space.notify_all();
        }
        drop(st);
        r.map(Ready::complete)
    }

    pub(crate) fn stats(&self) -> OutputStats {
        self.state.lock().stats.clone()
    }

    pub(crate) fn close(&self) {
        let mut st = self.state.lock();
        st.closed = true;
        self.cv_ready.notify_all();
        self.cv_space.notify_all();
        self.cv_flush.notify_all();
    }
}

fn find_filling(st: &mut State, id: u64) -> Option<&mut Filling> {
    if st.filling.as_ref().is_some_and(|f| f.id == id) {
        return st.filling.as_mut();
    }
    st.sealing.iter_mut().find(|f| f.id == id)
}

fn take_filling(st: &mut State, id: u64) -> Option<Filling> {
    if st.filling.as_ref().is_some_and(|f| f.id == id) {
        return st.filling.take();
    }
    let pos = st.sealing.iter().position(|f| f.id == id)?;
    Some(st.sealing.swap_remove(pos))
}

/// Drops the oldest queued batch, accounting its images as dropped.
fn drop_oldest(st: &mut State) -> bool {
    match st.ready.pop_front() {
        Some(r) => {
            // The batch buffer may only be recycled once its kernels finished.
            let b = r.complete();
            for it in b.items.iter().filter(|i| i.valid) {
                st.stats.dropped_images += 1;
                // Carry the drops this image was reporting over to the next one.
                *st.pending_drops.entry(it.meta.stream_id.clone()).or_default() +=
                    1 + it.meta.dropped_since_last;
            }
            st.stats.queued = st.ready.len();
            true
        }
        None => false,
    }
}

fn placeholder_meta() -> FrameMeta {
    FrameMeta {
        stream_id: Arc::from(""),
        seq: 0,
        pts: 0,
        keyframe: false,
        src_width: 0,
        src_height: 0,
        transform: camtensor_core::Affine2::IDENTITY,
        capture_time_ns: None,
        recv_time_ns: 0,
        ready_time_ns: 0,
        dropped_since_last: 0,
    }
}

fn flusher(out: std::sync::Weak<Output>) {
    loop {
        let Some(out) = out.upgrade() else { return };
        let mut st = out.state.lock();
        if st.closed {
            return;
        }
        let next = out.flush_due(&mut st);
        let wait = next.unwrap_or(Duration::from_millis(250));
        out.cv_flush.wait_for(&mut st, wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::AlignedBuf;
    use camtensor_core::{BatchConfig, DType};

    fn host_output(batch: BatchConfig, bp: Backpressure, depth: usize) -> Arc<Output> {
        let cfg = OutputConfig {
            tensor: TensorSpec { width: 4, height: 2, dtype: DType::U8, ..Default::default() },
            batch,
            backpressure: bp,
            queue_depth: depth,
            ..Default::default()
        };
        Output::new(
            "t",
            cfg,
            Device::Cpu,
            Box::new(|n| Ok(TensorStorage::Host(AlignedBuf::zeroed(n)))),
            Box::new(|addr, len| {
                // SAFETY: test buffers are host allocations of at least `len` bytes at `addr`.
                unsafe { std::ptr::write_bytes(addr as *mut u8, 0, len) };
                Ok(())
            }),
        )
        .unwrap()
    }

    fn meta(stream: &Arc<str>, seq: u64) -> FrameMeta {
        FrameMeta { stream_id: stream.clone(), seq, ..placeholder_meta() }
    }

    fn put(out: &Output, stream: &Arc<str>, seq: u64, value: u8) -> bool {
        out.submit(stream, |slot| {
            // SAFETY: slot points into a host batch buffer reserved for us.
            unsafe { std::ptr::write_bytes(slot.addr as *mut u8, value, slot.len) };
            Ok((meta(stream, seq), None, None))
        })
        .unwrap()
    }

    #[test]
    fn batches_of_max_size() {
        let out =
            host_output(BatchConfig { max_size: 3, ..Default::default() }, Backpressure::Block, 4);
        let s: Arc<str> = "cam".into();
        for i in 0..6 {
            assert!(put(&out, &s, i, i as u8 + 1));
        }
        let b = out.recv(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(b.tensor().shape(), [3, 3, 2, 4]);
        let bytes = b.tensor().as_bytes().unwrap();
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[24], 2);
        assert_eq!(bytes[48], 3);
        let b2 = out.recv(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(b2.items()[0].meta.seq, 3);
        assert!(out.try_recv().is_none());
    }

    #[test]
    fn deadline_flushes_partial_batch() {
        let out = host_output(
            BatchConfig { max_size: 8, max_wait_ms: 20, pad_to_max: false },
            Backpressure::DropOldest,
            4,
        );
        let s: Arc<str> = "cam".into();
        put(&out, &s, 0, 9);
        put(&out, &s, 1, 9);
        let t0 = Instant::now();
        let b = out.recv(Some(Duration::from_secs(2))).unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(5));
        assert_eq!(b.len(), 2);
        assert_eq!(b.tensor().shape()[0], 2);
    }

    #[test]
    fn pad_to_max_marks_invalid_slots() {
        let out = host_output(
            BatchConfig { max_size: 4, max_wait_ms: 10, pad_to_max: true },
            Backpressure::DropOldest,
            4,
        );
        let s: Arc<str> = "cam".into();
        put(&out, &s, 0, 7);
        let b = out.recv(Some(Duration::from_secs(2))).unwrap();
        assert_eq!(b.len(), 4);
        assert_eq!(b.valid_count(), 1);
        assert!(!b.items()[3].valid);
        assert!(b.tensor().as_bytes().unwrap()[24..].iter().all(|v| *v == 0));
    }

    #[test]
    fn drop_oldest_bounds_queue_and_reports_drops() {
        let out = host_output(BatchConfig::default(), Backpressure::DropOldest, 2);
        let s: Arc<str> = "cam".into();
        for i in 0..10 {
            put(&out, &s, i, 1);
        }
        let st = out.stats();
        assert_eq!(st.queued, 2);
        assert_eq!(st.dropped_images, 8);
        let a = out.recv(None).unwrap();
        let b = out.recv(None).unwrap();
        assert_eq!((a.items()[0].meta.seq, b.items()[0].meta.seq), (8, 9));
        put(&out, &s, 10, 1);
        let c = out.recv(None).unwrap();
        // Every dropped image is reported exactly once by a delivered image.
        let reported: u64 = [&a, &b, &c].iter().map(|x| x.items()[0].meta.dropped_since_last).sum();
        assert_eq!(reported, 8);
    }

    #[test]
    fn latest_only_keeps_one() {
        let out = host_output(BatchConfig::default(), Backpressure::LatestOnly, 4);
        let s: Arc<str> = "cam".into();
        for i in 0..5 {
            put(&out, &s, i, 1);
        }
        assert_eq!(out.stats().queued, 1);
        assert_eq!(out.recv(None).unwrap().items()[0].meta.seq, 4);
    }

    #[test]
    fn held_buffers_are_not_reused() {
        let out = host_output(BatchConfig::default(), Backpressure::DropOldest, 1);
        let s: Arc<str> = "cam".into();
        put(&out, &s, 0, 1);
        let held = out.recv(None).unwrap();
        for i in 1..20 {
            put(&out, &s, i, 2);
        }
        // The batch we hold still sees its own data.
        assert!(held.tensor().as_bytes().unwrap().iter().all(|v| *v == 1));
        assert!(out.stats().buffers <= 3);
    }

    #[test]
    fn concurrent_writers() {
        let out = host_output(
            BatchConfig { max_size: 4, max_wait_ms: 5, pad_to_max: false },
            Backpressure::Block,
            8,
        );
        let mut handles = Vec::new();
        for w in 0..4 {
            let out = out.clone();
            handles.push(std::thread::spawn(move || {
                let s: Arc<str> = format!("cam{w}").into();
                for i in 0..50 {
                    put(&out, &s, i, w as u8);
                }
            }));
        }
        let consumer = {
            let out = out.clone();
            std::thread::spawn(move || {
                let mut n = 0;
                while n < 200 {
                    let b = out.recv(Some(Duration::from_secs(5))).unwrap();
                    for (i, it) in b.items().iter().enumerate() {
                        assert!(it.valid);
                        let w: u8 = it.meta.stream_id[3..].parse().unwrap();
                        let img = &b.tensor().as_bytes().unwrap()[i * 24..(i + 1) * 24];
                        assert!(img.iter().all(|v| *v == w));
                    }
                    n += b.valid_count();
                }
                n
            })
        };
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(consumer.join().unwrap(), 200);
    }
}
