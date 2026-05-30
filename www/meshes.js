// Procedural geometry. Each object in the scene descriptor gets its own mesh
// whose dimensions already match the Rapier collider, so the rigid (rotation +
// translation only) model matrix from WASM positions it without any scaling.
//
// Vertex layout: interleaved [px, py, pz, nx, ny, nz] (stride 6 floats).
// Boxes use per-face normals (flat faces); wheel cylinders use smooth side
// normals (curved) + flat caps. Lighting is computed per-vertex (Gouraud).

// Build line indices (for wireframe) from triangle indices: each triangle's
// three edges become line segments. Overlapping edges overdraw harmlessly.
function lineIndicesFromTris(tri) {
  const lines = new Uint16Array(tri.length * 2);
  let o = 0;
  for (let i = 0; i < tri.length; i += 3) {
    const a = tri[i], b = tri[i + 1], c = tri[i + 2];
    lines[o++] = a; lines[o++] = b;
    lines[o++] = b; lines[o++] = c;
    lines[o++] = c; lines[o++] = a;
  }
  return lines;
}

function box(hx, hy, hz) {
  // 6 faces x 4 verts, outward normals.
  const faces = [
    // [normal], [4 corner offsets as (sx,sy,sz) in {-1,1}]
    { n: [1, 0, 0], c: [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]] },
    { n: [-1, 0, 0], c: [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, -1]] },
    { n: [0, 1, 0], c: [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]] },
    { n: [0, -1, 0], c: [[-1, -1, 1], [-1, -1, -1], [1, -1, -1], [1, -1, 1]] },
    { n: [0, 0, 1], c: [[1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]] },
    { n: [0, 0, -1], c: [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]] },
  ];
  const verts = [];
  const tris = [];
  let base = 0;
  for (const f of faces) {
    for (const [sx, sy, sz] of f.c) {
      verts.push(sx * hx, sy * hy, sz * hz, f.n[0], f.n[1], f.n[2]);
    }
    tris.push(base, base + 1, base + 2, base, base + 2, base + 3);
    base += 4;
  }
  return { positions: verts, triIndices: tris };
}

function cylinder(radius, halfWidth, segments = 20) {
  // Axis along local Z; radius in the XY plane; spans z = -halfWidth..+halfWidth.
  const verts = [];
  const tris = [];
  // --- Side: two rings with smooth (outward radial) normals.
  const ringStart = 0;
  for (let side = 0; side < 2; side++) {
    const z = side === 0 ? -halfWidth : halfWidth;
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      const cx = Math.cos(a), sy = Math.sin(a);
      verts.push(cx * radius, sy * radius, z, cx, sy, 0);
    }
  }
  for (let i = 0; i < segments; i++) {
    const i0 = ringStart + i;
    const i1 = ringStart + ((i + 1) % segments);
    const j0 = ringStart + segments + i;
    const j1 = ringStart + segments + ((i + 1) % segments);
    // Wound CCW as seen from outside (radial-outward normal faces the viewer).
    tris.push(i0, j1, j0, i0, i1, j1);
  }
  // --- Caps (flat normals along ±Z): center + fan.
  for (let side = 0; side < 2; side++) {
    const z = side === 0 ? -halfWidth : halfWidth;
    const nz = side === 0 ? -1 : 1;
    const center = verts.length / 6;
    verts.push(0, 0, z, 0, 0, nz);
    const rimStart = verts.length / 6;
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      verts.push(Math.cos(a) * radius, Math.sin(a) * radius, z, 0, 0, nz);
    }
    for (let i = 0; i < segments; i++) {
      const r0 = rimStart + i;
      const r1 = rimStart + ((i + 1) % segments);
      if (side === 0) tris.push(center, r1, r0);
      else tris.push(center, r0, r1);
    }
  }
  return { positions: verts, triIndices: tris };
}

// Build a terrain mesh from shared vertices/indices (generated in WASM so the
// collider and visuals match exactly). Per-vertex normals are accumulated from
// adjacent faces and normalized -> smooth (Gouraud) rocky shading.
function buildTerrain(desc) {
  const pos = desc.vertices; // flat [x,y,z,...]
  const idx = desc.indices; // flat triangle indices
  const vCount = pos.length / 3;
  const normals = new Float32Array(pos.length);

  for (let t = 0; t < idx.length; t += 3) {
    const ia = idx[t] * 3, ib = idx[t + 1] * 3, ic = idx[t + 2] * 3;
    const e1x = pos[ib] - pos[ia], e1y = pos[ib + 1] - pos[ia + 1], e1z = pos[ib + 2] - pos[ia + 2];
    const e2x = pos[ic] - pos[ia], e2y = pos[ic + 1] - pos[ia + 1], e2z = pos[ic + 2] - pos[ia + 2];
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }

  const interleaved = new Float32Array(vCount * 6);
  for (let v = 0; v < vCount; v++) {
    let nx = normals[v * 3], ny = normals[v * 3 + 1], nz = normals[v * 3 + 2];
    const len = Math.hypot(nx, ny, nz) || 1;
    interleaved[v * 6] = pos[v * 3];
    interleaved[v * 6 + 1] = pos[v * 3 + 1];
    interleaved[v * 6 + 2] = pos[v * 3 + 2];
    interleaved[v * 6 + 3] = nx / len;
    interleaved[v * 6 + 4] = ny / len;
    interleaved[v * 6 + 5] = nz / len;
  }

  const triIndices = new Uint16Array(idx);
  const lineIndices = lineIndicesFromTris(triIndices);
  return {
    interleaved,
    triIndices,
    lineIndices,
    color: desc.color,
    triCount: triIndices.length / 3,
  };
}

// Build a render mesh from one descriptor entry.
export function buildMesh(desc) {
  if (desc.kind === "terrain") {
    return buildTerrain(desc);
  }
  let geo;
  if (desc.kind === "box") {
    geo = box(desc.hx, desc.hy, desc.hz);
  } else if (desc.kind === "cylinder") {
    geo = cylinder(desc.radius, desc.halfWidth);
  } else {
    throw new Error("unknown mesh kind: " + desc.kind);
  }
  const positions = new Float32Array(geo.positions); // interleaved pos+normal
  const triIndices = new Uint16Array(geo.triIndices);
  const lineIndices = lineIndicesFromTris(triIndices);
  return {
    interleaved: positions,
    triIndices,
    lineIndices,
    color: desc.color,
    triCount: triIndices.length / 3,
  };
}
