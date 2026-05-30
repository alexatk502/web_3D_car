#!/usr/bin/env bash
# Compile the Rust physics core to WASM and emit the JS bindings into www/pkg.
set -euo pipefail
cd "$(dirname "$0")"
wasm-pack build --target web --out-dir www/pkg "$@"
echo "Built www/pkg. Run ./serve.sh and open http://localhost:8080"
