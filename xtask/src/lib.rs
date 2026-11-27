//! Developer tasks for camtensor (PTX generation, test media).
//!
//! ```text
//! cargo run -p xtask -- ptx           # regenerate crates/camtensor-cuda/kernels/*.ptx
//! cargo run -p xtask -- ptx --check   # fail if the committed PTX is stale
//! ```

pub mod ptx;

use std::path::PathBuf;

/// Workspace root (parent of the `xtask` directory).
pub fn workspace_root() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.parent().map(PathBuf::from).unwrap_or(dir)
}

/// Runs the task named by `args[0]`.
pub fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("ptx") => ptx::run(&args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            println!(
                "usage: cargo run -p xtask -- <task>\n\ntasks:\n  ptx [--check]   regenerate (or verify) the committed PTX kernels with NVRTC"
            );
            Ok(())
        }
        Some(other) => Err(format!("unknown task '{other}' (try 'help')")),
    }
}
