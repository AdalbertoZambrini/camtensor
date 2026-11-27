//! Tiny recycling pool for host frame buffers.

use std::sync::Arc;

/// Keeps a reference to the last few emitted buffers; once every consumer
/// has dropped its [`Arc`] (strong count back to 1) the allocation is reused
/// for the next frame instead of hitting the allocator again.
#[derive(Debug)]
pub(crate) struct BufPool {
    bufs: Vec<Arc<Vec<u8>>>,
    cap: usize,
}

impl BufPool {
    pub(crate) fn new(cap: usize) -> Self {
        Self { bufs: Vec::with_capacity(cap), cap }
    }

    /// Returns a buffer of exactly `len` bytes. Contents are unspecified
    /// (recycled buffers keep old pixels); callers overwrite all of it.
    pub(crate) fn take(&mut self, len: usize) -> Vec<u8> {
        if let Some(i) = self.bufs.iter().position(|b| Arc::strong_count(b) == 1)
            && let Ok(mut v) = Arc::try_unwrap(self.bufs.swap_remove(i))
        {
            v.resize(len, 0);
            return v;
        }
        vec![0u8; len]
    }

    /// Wraps `buf` in an `Arc`, remembering it for later reuse.
    pub(crate) fn share(&mut self, buf: Vec<u8>) -> Arc<Vec<u8>> {
        let arc = Arc::new(buf);
        if self.bufs.len() >= self.cap {
            // Forget the oldest entry (still in use by a consumer).
            self.bufs.remove(0);
        }
        self.bufs.push(Arc::clone(&arc));
        arc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recycles_released_buffers() {
        let mut pool = BufPool::new(2);
        let v = pool.take(16);
        let a = pool.share(v);
        let ptr = a.as_ptr();
        drop(a);
        let b = pool.take(16);
        assert_eq!(b.as_ptr(), ptr);
        assert_eq!(b.len(), 16);
    }

    #[test]
    fn does_not_recycle_buffers_in_use() {
        let mut pool = BufPool::new(2);
        let a = pool.share(vec![1; 8]);
        let b = pool.take(8);
        assert_ne!(b.as_ptr(), a.as_ptr());
    }
}
