#!/usr/bin/env bash
# Serve the app over http (localhost is a secure context, so WebGPU works).
# Opening index.html as a file:// URL will NOT work (ES modules + WASM fetch).
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8080}"
echo "Serving on http://localhost:${PORT}  (Ctrl+C to stop)"
exec python3 -m http.server "${PORT}" --directory www
