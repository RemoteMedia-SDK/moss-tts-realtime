//! MOSS-TTS-Realtime as a single-file loadable Python plugin.
//!
//! Canonical Python-plugin example — proof that
//! `remotemedia_plugin_sdk::python_plugin_export!` handles non-trivial
//! plugins with vendored sub-packages, streaming async generators,
//! multi-output emission, aux-port envelopes, and heavy PEP 723 deps
//! (transformers, torch, torchaudio, accelerate, …).
//!
//! Companion to:
//! - [`examples/echo-python-loadable/`] — minimum-viable Python plugin
//!   (zero deps, single file).
//! - [`examples/silero-vad-loadable/`] — Rust-side equivalent (in-tree
//!   node shipped as a cdylib via `#[node(loadable_export)]`).
//!
//! ## What this validates today
//! - Vendored sub-package (`_vendor/mossttsrealtime/`) survives
//!   `include_dir!` embedding alongside the primary module.
//! - PEP 723 deps parser extracts the realistic mixed list from
//!   `moss_tts_realtime.py` (`transformers>=5.0,<6.0`, `torch>=2.1`,
//!   `torchaudio>=2.1`, `accelerate>=0.33`, `safetensors<0.5`,
//!   `torchcodec`).
//! - `cargo build --release` produces one `.so`/`.dylib`/`.dll`.
//! - Host's `LoadableNodeBundle::load()` registers the factory under
//!   `node_type = "MossTTSRealtimeNode"`.
//! - First-load provisioning (Task 3.3.3) runs through
//!   `PythonEnvManager::ensure_env(&deps)` against the parsed list.
//!   First-run venv install is multi-GB and several minutes (torch);
//!   subsequent runs reuse the uv-managed cache.
//!
//! ## What's deferred (Task 3.3.4-3.3.6)
//! - Subprocess spawn + iceoryx2 control/input/output channels.
//! - Round-trip: text delta → audio chunks + `<|audio_end|>` marker.
//! - Aux-port flow (`audio.in.reference`, `audio.in.reset`, …).
//!
//! Until that lands, factory `create()` returns a precise RErr
//! including the resolved python_executable path — the concrete handoff
//! for the next session.
//!
//! ## Embedded source freshness
//! `embedded/moss_tts_realtime.py` and `embedded/_vendor/mossttsrealtime/`
//! are byte-for-byte copies of the in-tree sources at
//! `clients/python/remotemedia/nodes/ml/moss_tts_realtime.py` and
//! `clients/python/remotemedia/_vendor/mossttsrealtime/` respectively.
//! Re-sync when the in-tree sources change:
//!
//! ```bash
//! cp clients/python/remotemedia/nodes/ml/moss_tts_realtime.py \
//!    examples/moss-tts-realtime-loadable/embedded/
//! cp clients/python/remotemedia/_vendor/mossttsrealtime/*.py \
//!    examples/moss-tts-realtime-loadable/embedded/_vendor/mossttsrealtime/
//! ```

use include_dir::{include_dir, Dir};

/// Embedded Python source tree — primary module + vendored
/// `mossttsrealtime` package. Resolved at compile time relative to
/// `$CARGO_MANIFEST_DIR`. Build fails (loudly) if either is missing.
static EMBED: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/embedded");

remotemedia_plugin_sdk::python_plugin_export! {
    node_type: "MossTTSRealtimeNode",
    module:    "moss_tts_realtime",
    class:     "MossTTSRealtimeNode",
    embedded:  &EMBED,
}
