//! Client for the camtensor service data plane.
//!
//! The service delivers batches through a shared-memory ring (one file per
//! subscription, mapped by both processes) and a small newline-delimited
//! JSON control channel over TCP. Receiving a batch costs one memcpy on the
//! service side and zero copies on the client side.
//!
//! ```no_run
//! let mut client = camtensor_client::Client::connect("127.0.0.1:7071", "detector")?;
//! loop {
//!     let batch = client.recv()?;
//!     println!("{} images {:?} {}", batch.items.len(), batch.shape, batch.dtype);
//!     let _bytes: &[u8] = batch.data(); // released when `batch` is dropped
//! }
//! # Ok::<(), camtensor_core::Error>(())
//! ```

#![warn(missing_docs)]

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use camtensor_core::{Error, FrameMeta, Result, TensorSpec};

pub mod protocol {
    //! Wire protocol (newline-delimited JSON) shared with the service.
    use super::*;

    /// Messages sent by the client.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "op", rename_all = "snake_case")]
    pub enum ClientMsg {
        /// Subscribe to an output.
        Subscribe {
            /// Output name.
            output: String,
            /// Number of shared-memory slots (batches in flight).
            #[serde(default = "default_slots")]
            slots: usize,
        },
        /// The client is done with a slot.
        Release {
            /// Slot index.
            slot: usize,
        },
    }

    fn default_slots() -> usize {
        4
    }

    /// Messages sent by the service.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "op", rename_all = "snake_case")]
    pub enum ServerMsg {
        /// Subscription accepted.
        Subscribed {
            /// Output name.
            output: String,
            /// Path of the shared-memory file to map.
            shm_path: String,
            /// Number of slots.
            slots: usize,
            /// Bytes per slot (`max_batch * image_bytes`).
            slot_bytes: usize,
            /// Image format.
            tensor: TensorSpec,
            /// Maximum batch size.
            max_batch: usize,
        },
        /// A batch was written to `slot`.
        Batch {
            /// Slot index.
            slot: usize,
            /// Output batch counter.
            seq: u64,
            /// Tensor shape.
            shape: [usize; 4],
            /// Element type (`u8`, `f16`, `f32`).
            dtype: String,
            /// Bytes written in the slot.
            bytes: usize,
            /// Per-image metadata.
            items: Vec<RemoteItem>,
        },
        /// Fatal error; the connection is closed afterwards.
        Error {
            /// Description.
            message: String,
        },
    }

    /// Metadata of one batch slot.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteItem {
        /// `false` for padding slots.
        pub valid: bool,
        /// Image metadata.
        pub meta: FrameMeta,
    }
}

pub use protocol::RemoteItem;
use protocol::{ClientMsg, ServerMsg};

/// Connection to one output of a camtensor service.
#[derive(Debug)]
pub struct Client {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    map: Mmap,
    /// Output name.
    pub output: String,
    /// Image format of the output.
    pub tensor: TensorSpec,
    /// Maximum batch size.
    pub max_batch: usize,
    slot_bytes: usize,
    line: String,
}

impl Client {
    /// Connects to the data port of a service and subscribes to `output`.
    pub fn connect(addr: &str, output: &str) -> Result<Self> {
        Self::connect_with_slots(addr, output, 4)
    }

    /// Like [`Client::connect`] with an explicit number of slots.
    pub fn connect_with_slots(addr: &str, output: &str, slots: usize) -> Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        let mut writer = stream.try_clone()?;
        let mut reader = BufReader::new(stream);
        send(&mut writer, &ClientMsg::Subscribe { output: output.into(), slots })?;
        let mut line = String::new();
        match read_msg(&mut reader, &mut line)? {
            ServerMsg::Subscribed { output, shm_path, slot_bytes, tensor, max_batch, .. } => {
                let file = std::fs::OpenOptions::new().read(true).open(&shm_path)?;
                // SAFETY: the service keeps the file alive and only writes a
                // slot before announcing it and never while we hold it.
                let map = unsafe { Mmap::map(&file)? };
                Ok(Self { reader, writer, map, output, tensor, max_batch, slot_bytes, line })
            }
            ServerMsg::Error { message } => Err(Error::config(message)),
            other => Err(Error::Ingest(format!("unexpected message {other:?}"))),
        }
    }

    /// Waits for the next batch. The slot is released when the returned
    /// [`RemoteBatch`] is dropped.
    pub fn recv(&mut self) -> Result<RemoteBatch<'_>> {
        match read_msg(&mut self.reader, &mut self.line)? {
            ServerMsg::Batch { slot, seq, shape, dtype, bytes, items } => {
                if bytes > self.slot_bytes || (slot + 1) * self.slot_bytes > self.map.len() {
                    return Err(Error::Ingest("batch outside shared memory".into()));
                }
                Ok(RemoteBatch { client: self, slot, seq, shape, dtype, bytes, items })
            }
            ServerMsg::Error { message } => Err(Error::Ingest(message)),
            other => Err(Error::Ingest(format!("unexpected message {other:?}"))),
        }
    }
}

/// A batch living in shared memory.
#[derive(Debug)]
pub struct RemoteBatch<'a> {
    client: &'a mut Client,
    slot: usize,
    /// Batch counter.
    pub seq: u64,
    /// Tensor shape.
    pub shape: [usize; 4],
    /// Element type (`u8`, `f16`, `f32`).
    pub dtype: String,
    bytes: usize,
    /// Per-image metadata.
    pub items: Vec<RemoteItem>,
}

impl RemoteBatch<'_> {
    /// Raw tensor bytes (contiguous, native endianness).
    pub fn data(&self) -> &[u8] {
        let off = self.slot * self.client.slot_bytes;
        &self.client.map[off..off + self.bytes]
    }

    /// Typed view for `f32` batches.
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != "f32" {
            return None;
        }
        let d = self.data();
        if d.as_ptr() as usize % 4 != 0 {
            return None;
        }
        // SAFETY: alignment checked, length is a multiple of 4 by construction,
        // and every bit pattern is a valid f32.
        Some(unsafe { std::slice::from_raw_parts(d.as_ptr() as *const f32, d.len() / 4) })
    }
}

impl Drop for RemoteBatch<'_> {
    fn drop(&mut self) {
        let _ = send(&mut self.client.writer, &ClientMsg::Release { slot: self.slot });
    }
}

fn send(w: &mut TcpStream, msg: &ClientMsg) -> Result<()> {
    let mut s = serde_json::to_string(msg).map_err(|e| Error::Ingest(e.to_string()))?;
    s.push('\n');
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn read_msg(r: &mut BufReader<TcpStream>, line: &mut String) -> Result<ServerMsg> {
    line.clear();
    if r.read_line(line)? == 0 {
        return Err(Error::Closed);
    }
    serde_json::from_str(line.trim_end()).map_err(|e| Error::Ingest(format!("bad message: {e}")))
}

#[cfg(test)]
mod tests {
    use super::protocol::*;

    #[test]
    fn protocol_roundtrip() {
        let m = ClientMsg::Subscribe { output: "det".into(), slots: 3 };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"op":"subscribe","output":"det","slots":3}"#);
        let back: ClientMsg = serde_json::from_str(r#"{"op":"subscribe","output":"x"}"#).unwrap();
        assert!(matches!(back, ClientMsg::Subscribe { slots: 4, .. }));
        let r: ClientMsg = serde_json::from_str(r#"{"op":"release","slot":2}"#).unwrap();
        assert!(matches!(r, ClientMsg::Release { slot: 2 }));
    }
}
