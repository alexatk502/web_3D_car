#!/usr/bin/env bash
# Compile the Rust physics core to threaded WASM and emit JS bindings into www/pkg.
#
# Threading needs: nightly (pinned via rust-toolchain.toml), the atomics/SIMD
# target features (.cargo/config.toml), and `-Z build-std` to recompile std with
# them. wasm-bindgen-rayon then provides the Web Worker thread pool. The app must
# be served cross-origin-isolated (see serve.py) for SharedArrayBuffer to work.
set -euo pipefail
cd "$(dirname "$0")"
# build-std passed via env (wasm-pack 0.13 mis-parses `-- -Z`); cargo reads
# CARGO_UNSTABLE_* on nightly.
CARGO_UNSTABLE_BUILD_STD="std,panic_abort" \
  wasm-pack build --target web --out-dir www/pkg "$@"
echo "Built www/pkg (threaded). Run ./serve.py (NOT serve.sh) for COOP/COEP headers."
