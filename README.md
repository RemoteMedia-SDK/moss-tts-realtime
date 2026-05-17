# moss-tts-realtime — MOSS-TTS-Realtime as a RemoteMedia SDK loadable plugin

Canonical Path 4 Python plugin: a Rust cdylib that embeds the full
Python implementation + vendored `mossttsrealtime` sub-package via
`include_dir!`, provisions a managed uv venv on first load from its
`@python_requires([...])` declaration, and runs the model in a
subprocess speaking iceoryx2 shared-memory IPC.

> **Heavy plugin.** First-load provisioning installs `transformers`,
> `torch`, `torchaudio`, `accelerate`, `safetensors`, `torchcodec` —
> multi-GB, several minutes. Subsequent loads hit the uv cache and
> spawn instantly. Plan accordingly.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["moss-tts-realtime@v0.1.0"],
  "nodes": [
    {
      "id": "tts",
      "node_type": "MossTTSRealtimeNode",
      "params": {}
    }
  ]
}
```

The SDK resolver expands `moss-tts-realtime@v0.1.0` to
`github.com/RemoteMedia-SDK/moss-tts-realtime`, fetches `plugin.toml`,
then falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset. Note: because
`language="rust"` in `plugin.toml`, the resolver picks the cdylib
asset path — NOT the Python source-load path. The Python inside is an
implementation detail of the cdylib.

> **Status:** plugin.toml + source published. **Prebuilt release
> binaries are not yet uploaded** — the matrix-build CI workflow is
> pending. Until then, consumers should either build the cdylib
> themselves (see below) or use a local-path plugin entry.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/moss-tts-realtime
cd moss-tts-realtime
cargo build --release
# → target/release/libmoss_tts_realtime_loadable_plugin.so
```

Then reference it from your manifest:

```json
{ "plugins": ["./path/to/libmoss_tts_realtime_loadable_plugin.so"] }
```

## What it exports

| Node type             | Input         | Output                                              |
|-----------------------|---------------|-----------------------------------------------------|
| `MossTTSRealtimeNode` | Text (deltas) | Audio chunks + `<\|audio_end\|>` marker (aux ports) |

Streaming multi-output: the node emits audio chunks as they're
generated, plus aux-port envelopes for `audio.in.reference`,
`audio.in.reset`, and the end-of-audio marker.

## What's in the repo

```
moss-tts-realtime/
├── plugin.toml                              ← metadata
├── Cargo.toml                               ← git-deps the SDK at a pinned rev
├── src/lib.rs                               ← `python_plugin_export!{...}` macro call
├── embedded/
│   ├── moss_tts_realtime.py                 ← the actual node implementation
│   └── _vendor/mossttsrealtime/             ← vendored upstream package
├── run.sh                                   ← local smoke-test driver
└── README.md
```

`embedded/` is byte-for-byte mirrored from the monorepo's
`clients/python/remotemedia/{nodes/ml/moss_tts_realtime.py,_vendor/mossttsrealtime/}`.
Re-sync when the in-tree sources change.

## License

See `LICENSE.md`. This plugin reuses RemoteMedia SDK source and is
governed by the same RemoteMedia SDK Community License 1.0.
