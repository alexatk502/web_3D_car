// Minimal column-major mat4 helpers. The WASM side produces the view matrix and
// per-object model matrices; JS builds the (backend-specific) projection and the
// view-projection product.

// Perspective projection. `zeroToOne` selects clip-space Z range:
//   true  -> [0, 1]  (WebGPU / D3D / Metal / Vulkan)
//   false -> [-1, 1] (WebGL / OpenGL)
export function perspective(fovYRad, aspect, near, far, zeroToOne) {
  const f = 1.0 / Math.tan(fovYRad / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[11] = -1;
  if (zeroToOne) {
    out[10] = far / (near - far);
    out[14] = (far * near) / (near - far);
  } else {
    const nf = 1 / (near - far);
    out[10] = (far + near) * nf;
    out[14] = 2 * far * near * nf;
  }
  return out;
}

// C = A * B, all column-major 4x4. C[col*4+row] = sum_k A[k*4+row] * B[col*4+k].
export function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = s;
    }
  }
  return out;
}

export function deg2rad(d) {
  return (d * Math.PI) / 180;
}
