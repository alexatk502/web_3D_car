// Picks the rendering backend: WebGPU if available, otherwise WebGL2. Both
// expose the same interface, so the game loop is backend-agnostic.

import { WebGPUBackend } from "./webgpu_backend.js";
import { WebGLBackend } from "./webgl_backend.js";

// `force` can be "webgpu" or "webgl" to override detection (used by the UI to
// demonstrate the fallback path); otherwise auto-detect.
export async function createRenderer(canvas, force) {
  if (force !== "webgl" && navigator.gpu) {
    try {
      const backend = new WebGPUBackend();
      await backend.init(canvas);
      return { backend, name: "WebGPU", zeroToOne: true };
    } catch (e) {
      console.warn("WebGPU init failed, falling back to WebGL2:", e);
    }
  }
  const backend = new WebGLBackend();
  await backend.init(canvas);
  return { backend, name: "WebGL2", zeroToOne: false };
}
