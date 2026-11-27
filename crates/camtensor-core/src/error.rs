use std::fmt;

/// Result alias used across camtensor.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Error type shared by all camtensor crates.
///
/// Variants are coarse on purpose: they tell the caller *which stage* failed
/// and whether retrying makes sense. Details go in the message.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Invalid configuration or argument supplied by the user.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// Network / RTSP level failure. Usually transient; the engine reconnects.
    #[error("ingest error: {0}")]
    Ingest(String),
    /// The bitstream could not be parsed.
    #[error("bitstream error: {0}")]
    Bitstream(String),
    /// Decoder failure.
    #[error("decode error: {0}")]
    Decode(String),
    /// Preprocessing failure.
    #[error("preprocess error: {0}")]
    Preprocess(String),
    /// CUDA / NVDEC failure.
    #[error("cuda error: {0}")]
    Cuda(String),
    /// A required runtime library (driver, libavcodec, ...) could not be loaded.
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    /// Operation timed out.
    #[error("timed out")]
    Timeout,
    /// The engine, stream or output was closed.
    #[error("closed")]
    Closed,
    /// Requested item does not exist.
    #[error("not found: {0}")]
    NotFound(String),
    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Shorthand constructor for [`Error::Config`].
    pub fn config(msg: impl fmt::Display) -> Self {
        Self::Config(msg.to_string())
    }
    /// Shorthand constructor for [`Error::Decode`].
    pub fn decode(msg: impl fmt::Display) -> Self {
        Self::Decode(msg.to_string())
    }
    /// Shorthand constructor for [`Error::Cuda`].
    pub fn cuda(msg: impl fmt::Display) -> Self {
        Self::Cuda(msg.to_string())
    }
    /// Shorthand constructor for [`Error::Unavailable`].
    pub fn unavailable(msg: impl fmt::Display) -> Self {
        Self::Unavailable(msg.to_string())
    }
}
