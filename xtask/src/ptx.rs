//! `ptx` task: compiles `crates/camtensor-cuda/kernels/*.cu` to PTX with a
//! dynamically loaded NVRTC.
//!
//! NVRTC is located through, in order:
//! 1. `CAMTENSOR_NVRTC` (path to the NVRTC library or to its directory);
//! 2. `.devdeps/nvrtc12/nvidia/cuda_nvrtc/{bin,lib}` in any ancestor of the
//!    workspace (the pip `nvidia-cuda-nvrtc-cu12==12.0.*` wheel layout);
//! 3. the platform library search path (`PATH` / `LD_LIBRARY_PATH`).
//!
//! The kernels target `compute_52` so that the PTX JIT-compiles on every
//! GPU from Maxwell to the newest one; NVRTC 12.0 is preferred because the
//! PTX ISA version it emits (8.0) is accepted by any driver >= 525.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;

type NvrtcResult = c_int;
type NvrtcProgram = *mut c_void;

/// Kernels to build: (source file, PTX file).
const KERNELS: &[(&str, &str)] = &[("preprocess.cu", "preprocess.ptx")];

/// NVRTC options used for every kernel.
const OPTIONS: &[&str] = &["--gpu-architecture=compute_52", "--fmad=false", "--std=c++14"];

struct Nvrtc {
    _lib: Library,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult,
    error_string: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
    create: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    compile: unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    destroy: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
}

#[cfg(windows)]
const LIB_NAMES: &[&str] = &[
    "nvrtc64_120_0.dll",
    "nvrtc64_121_0.dll",
    "nvrtc64_122_0.dll",
    "nvrtc64_123_0.dll",
    "nvrtc64_124_0.dll",
    "nvrtc64_125_0.dll",
    "nvrtc64_126_0.dll",
    "nvrtc64_128_0.dll",
    "nvrtc64_129_0.dll",
    "nvrtc64_130_0.dll",
];
#[cfg(not(windows))]
const LIB_NAMES: &[&str] = &["libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"];

#[cfg(windows)]
const DEVDEPS_SUBDIR: &str = ".devdeps/nvrtc12/nvidia/cuda_nvrtc/bin";
#[cfg(not(windows))]
const DEVDEPS_SUBDIR: &str = ".devdeps/nvrtc12/nvidia/cuda_nvrtc/lib";

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("CAMTENSOR_NVRTC") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            out.extend(LIB_NAMES.iter().map(|n| p.join(n)));
        } else {
            out.push(p);
        }
    }
    let root = crate::workspace_root();
    for dir in root.ancestors() {
        let d = dir.join(DEVDEPS_SUBDIR);
        if d.is_dir() {
            out.extend(LIB_NAMES.iter().map(|n| d.join(n)));
        }
    }
    out.extend(LIB_NAMES.iter().map(PathBuf::from));
    out
}

fn load_library(path: &Path) -> Result<Library, libloading::Error> {
    // NVRTC loads its builtins library from the platform search path, so
    // make the directory of an explicitly located NVRTC searchable.
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        let var = if cfg!(windows) { "PATH" } else { "LD_LIBRARY_PATH" };
        let old = std::env::var_os(var).unwrap_or_default();
        let mut paths = vec![dir.to_path_buf()];
        paths.extend(std::env::split_paths(&old));
        if let Ok(joined) = std::env::join_paths(paths) {
            // SAFETY: xtask is single-threaded at this point; nothing reads
            // the environment concurrently.
            unsafe { std::env::set_var(var, joined) };
        }
    }
    // SAFETY: loading NVRTC runs no unusual initialisation code.
    unsafe { Library::new(path) }
}

impl Nvrtc {
    fn load() -> Result<Self, String> {
        let mut errors = Vec::new();
        for path in candidates() {
            if path.components().count() > 1 && !path.exists() {
                continue;
            }
            match load_library(&path) {
                Ok(lib) => match Self::from_library(lib) {
                    Ok(n) => {
                        eprintln!("using NVRTC from {}", path.display());
                        return Ok(n);
                    }
                    Err(e) => errors.push(format!("{}: {e}", path.display())),
                },
                Err(e) => errors.push(format!("{}: {e}", path.display())),
            }
        }
        Err(format!(
            "NVRTC not found; set CAMTENSOR_NVRTC to the NVRTC library \
             (e.g. nvrtc64_120_0.dll / libnvrtc.so.12). Tried:\n  {}",
            errors.join("\n  ")
        ))
    }

    fn from_library(lib: Library) -> Result<Self, String> {
        macro_rules! sym {
            ($name:literal) => {
                // SAFETY: the declared field types match the NVRTC C API.
                unsafe { *lib.get($name).map_err(|e| format!("{}: {e}", $name))? }
            };
        }
        Ok(Self {
            version: sym!("nvrtcVersion"),
            error_string: sym!("nvrtcGetErrorString"),
            create: sym!("nvrtcCreateProgram"),
            compile: sym!("nvrtcCompileProgram"),
            log_size: sym!("nvrtcGetProgramLogSize"),
            log: sym!("nvrtcGetProgramLog"),
            ptx_size: sym!("nvrtcGetPTXSize"),
            ptx: sym!("nvrtcGetPTX"),
            destroy: sym!("nvrtcDestroyProgram"),
            _lib: lib,
        })
    }

    fn check(&self, r: NvrtcResult, what: &str) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        // SAFETY: nvrtcGetErrorString returns a static string for any code.
        let s = unsafe { CStr::from_ptr((self.error_string)(r)) }.to_string_lossy();
        Err(format!("{what}: {s}"))
    }

    fn version_string(&self) -> String {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: valid out pointers.
        unsafe { (self.version)(&mut major, &mut minor) };
        format!("{major}.{minor}")
    }

    fn compile_to_ptx(&self, source: &str, name: &str) -> Result<String, String> {
        let src = CString::new(source).map_err(|_| "source contains NUL")?;
        let cname = CString::new(name).map_err(|_| "name contains NUL")?;
        let mut prog: NvrtcProgram = std::ptr::null_mut();
        // SAFETY: valid strings and out pointer; no headers.
        let r = unsafe {
            (self.create)(
                &mut prog,
                src.as_ptr(),
                cname.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        self.check(r, "nvrtcCreateProgram")?;
        let opts: Vec<CString> = OPTIONS.iter().map(|o| CString::new(*o).unwrap()).collect();
        let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        // SAFETY: `prog` is valid; option pointers outlive the call.
        let r = unsafe { (self.compile)(prog, opt_ptrs.len() as c_int, opt_ptrs.as_ptr()) };
        let log = self.read(prog, self.log_size, self.log).unwrap_or_default();
        let result = if r != 0 {
            let e = self.check(r, "nvrtcCompileProgram").err().unwrap_or_default();
            Err(format!("{e}\n{log}"))
        } else {
            if !log.trim().is_empty() {
                eprintln!("{name}: {log}");
            }
            self.read(prog, self.ptx_size, self.ptx)
        };
        // SAFETY: `prog` was created above and is destroyed once.
        unsafe { (self.destroy)(&mut prog) };
        result
    }

    fn read(
        &self,
        prog: NvrtcProgram,
        size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
        get: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    ) -> Result<String, String> {
        let mut n = 0usize;
        // SAFETY: valid program and out pointer.
        self.check(unsafe { size(prog, &mut n) }, "nvrtc size query")?;
        let mut buf = vec![0u8; n.max(1)];
        // SAFETY: the buffer holds the reported size (NUL included).
        self.check(unsafe { get(prog, buf.as_mut_ptr() as *mut c_char) }, "nvrtc read")?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }
}

/// Entry point of `xtask ptx [--check]`.
pub fn run(args: &[String]) -> Result<(), String> {
    let check = args.iter().any(|a| a == "--check");
    let nvrtc = Nvrtc::load()?;
    let version = nvrtc.version_string();
    let dir = crate::workspace_root().join("crates/camtensor-cuda/kernels");
    let mut stale = Vec::new();
    for (src_name, ptx_name) in KERNELS {
        let src_path = dir.join(src_name);
        let source = std::fs::read_to_string(&src_path)
            .map_err(|e| format!("{}: {e}", src_path.display()))?;
        let ptx = nvrtc.compile_to_ptx(&source, src_name)?;
        let ptx = ptx.replace("\r\n", "\n");
        let out = format!(
            "//\n// Generated by `cargo run -p xtask -- ptx` from {src_name}\n\
             // with NVRTC {version} ({}). Do not edit.\n{ptx}",
            OPTIONS.join(" ")
        );
        let ptx_path = dir.join(ptx_name);
        if check {
            let current =
                std::fs::read_to_string(&ptx_path).unwrap_or_default().replace("\r\n", "\n");
            if current != out {
                stale.push(ptx_path.display().to_string());
            }
        } else {
            std::fs::write(&ptx_path, out.as_bytes())
                .map_err(|e| format!("{}: {e}", ptx_path.display()))?;
            println!("wrote {} ({} bytes, NVRTC {version})", ptx_path.display(), out.len());
        }
    }
    if !stale.is_empty() {
        return Err(format!("stale PTX (run `cargo run -p xtask -- ptx`): {}", stale.join(", ")));
    }
    if check {
        println!("PTX is up to date");
    }
    Ok(())
}
