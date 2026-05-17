#!/usr/bin/env bash
# Build the plugin cdylib + the host binary, then run the host against
# the freshly-built plugin. Mirrors the shape of
# `examples/echo-python-loadable/run.sh` for the canonical example path.
#
# Pass `--exercise` to drive the provisioning step (installs torch +
# transformers — multi-GB, several minutes on first run).
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Building plugin cdylib (libmoss_tts_realtime_loadable_plugin.so)..."
cargo build

echo "==> Building host..."
(cd host && cargo build)

echo "==> Running host against the freshly-built plugin..."
exec ./host/target/debug/moss-tts-realtime-loadable-host \
  ./target/debug/libmoss_tts_realtime_loadable_plugin.so \
  "$@"
