//! Raw, hand-written FFI declarations for the CUDA driver API and the NVDEC
//! (`nvcuvid`) API, loaded at runtime.
//!
//! Struct layouts follow the MIT-licensed headers of FFmpeg's
//! `nv-codec-headers` (`dynlink_cuda.h`, `dynlink_cuviddec.h`,
//! `dynlink_nvcuvid.h`). Only what camtensor needs is declared. The layout
//! tests at the bottom of [`cuvid`] and [`driver`] pin sizes and offsets for
//! x86_64 Windows (LLP64) and Linux (LP64).
//!
//! Everything here is `unsafe` to use; prefer the safe wrappers in the
//! parent crate.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    missing_docs,
    missing_debug_implementations
)]

pub mod cuvid;
pub mod driver;

use libloading::Library;

/// Opens the first library in `names` that loads.
pub(crate) fn open_library(names: &[&str]) -> Result<Library, String> {
    let mut errors = Vec::new();
    for name in names {
        // SAFETY: loading a vendor driver library runs its initialisers; the
        // NVIDIA driver libraries are designed to be loaded this way and we
        // only ever load them by their well-known names.
        match unsafe { Library::new(*name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }
    Err(errors.join("; "))
}

/// Declares a struct of function pointers loaded from a [`Library`].
macro_rules! dynamic_api {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $( $field:ident = $sym:literal : fn($($arg:ty),* $(,)?) -> $ret:ty; )*
        }
    ) => {
        $(#[$meta])*
        pub struct $name {
            _lib: libloading::Library,
            $( pub $field: unsafe extern "system" fn($($arg),*) -> $ret, )*
        }

        impl $name {
            /// Resolves every symbol from `lib`.
            ///
            /// # Safety
            /// `lib` must be the library the declared signatures belong to.
            pub(crate) unsafe fn load(lib: libloading::Library) -> Result<Self, String> {
                $(
                    // SAFETY: the caller guarantees the signature matches the
                    // exported symbol; the function pointer is copied out of
                    // the `Symbol` and stays valid while `_lib` is alive,
                    // which it is for the lifetime of `Self`.
                    let $field = unsafe {
                        *lib.get::<unsafe extern "system" fn($($arg),*) -> $ret>($sym)
                            .map_err(|e| format!("missing symbol {}: {e}", $sym))?
                    };
                )*
                Ok(Self { _lib: lib, $($field),* })
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    };
}

pub(crate) use dynamic_api;
