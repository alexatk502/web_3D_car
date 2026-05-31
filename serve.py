#!/usr/bin/env python3
"""Dev server that sets the cross-origin-isolation headers threaded WASM needs.

SharedArrayBuffer (and therefore the wasm-bindgen-rayon thread pool) is only
available when the page is *cross-origin isolated*, which requires:
    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp
`python3 -m http.server` does not send these, so the worker pool silently fails
to start. Run this instead:  ./serve.py [port]   (default 8080)
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # WASM must be served with the right type for streaming compilation.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    handler = partial(Handler, directory="www")
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"Serving www/ cross-origin-isolated on http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
