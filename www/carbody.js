// Procedural car-body mesh + nearest-node skinning.
//
// The body hull is authored as subdivided boxes (a lower body + a raised cabin)
// so its surface is dense enough to show local crumpling. Each body vertex is
// bound to the K nearest chassis nodes (inverse-square weights) plus a fixed
// rest offset, so as the chassis node grid deforms the body follows it locally —
// dents and crush zones appear on the panels, BeamNG-style.

const K = 4; // nodes each vertex is skinned to
const SEGS = 3; // subdivisions per box face (higher = finer crumple)

// Subdivided axis-aligned box in cage-param space. P(u,v,w) maps params -> local
// position. Returns {positions:[[x,y,z]...], tris:[[a,b,c]...]}. Culling is
// disabled when drawing the body, so winding is unimportant.
function boxFaces(P, lo, hi, segs) {
  const positions = [];
  const tris = [];
  // Each face: the two varying param-axes and the fixed one.
  const faces = [
    { fix: 0, val: lo[0], ua: 1, va: 2 },
    { fix: 0, val: hi[0], ua: 1, va: 2 },
    { fix: 1, val: lo[1], ua: 0, va: 2 },
    { fix: 1, val: hi[1], ua: 0, va: 2 },
    { fix: 2, val: lo[2], ua: 0, va: 1 },
    { fix: 2, val: hi[2], ua: 0, va: 1 },
  ];
  for (const f of faces) {
    const base = positions.length;
    for (let i = 0; i <= segs; i++) {
      for (let j = 0; j <= segs; j++) {
        const p = [0, 0, 0];
        p[f.fix] = f.val;
        p[f.ua] = lo[f.ua] + (hi[f.ua] - lo[f.ua]) * (i / segs);
        p[f.va] = lo[f.va] + (hi[f.va] - lo[f.va]) * (j / segs);
        positions.push(P(p[0], p[1], p[2]));
      }
    }
    const row = segs + 1;
    for (let i = 0; i < segs; i++) {
      for (let j = 0; j < segs; j++) {
        const a = base + i * row + j;
        const b = base + i * row + j + 1;
        const c = base + (i + 1) * row + j;
        const d = base + (i + 1) * row + j + 1;
        tris.push([a, b, d], [a, d, c]);
      }
    }
  }
  return { positions, tris };
}

function mergeBoxes(boxes) {
  const positions = [];
  const tris = [];
  for (const b of boxes) {
    const off = positions.length;
    for (const p of b.positions) positions.push(p);
    for (const t of b.tris) tris.push([t[0] + off, t[1] + off, t[2] + off]);
  }
  return { positions, tris };
}

// Author the hull in cage-param space (u,v,w in 0..1 over the body box; values
// may exceed [0,1] — the cabin roof rises above the chassis cage).
function carHull(min, max) {
  const lerp = (a, b, t) => a + (b - a) * t;
  const P = (u, v, w) => [
    lerp(min[0], max[0], u),
    lerp(min[1], max[1], v),
    lerp(min[2], max[2], w),
  ];
  const body = boxFaces(P, [0.0, -0.15, 0.0], [1.0, 0.55, 1.0], SEGS);
  const cabin = boxFaces(P, [0.30, 0.55, 0.12], [0.78, 1.35, 0.88], SEGS);
  return mergeBoxes([body, cabin]);
}

// --- small vector helpers ---
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

function centroidOf(pos, group) {
  let x = 0, y = 0, z = 0;
  for (const i of group) {
    x += pos[i * 3];
    y += pos[i * 3 + 1];
    z += pos[i * 3 + 2];
  }
  const n = group.length || 1;
  return [x / n, y / n, z / n];
}

// Orthonormal car frame [forward, right, up] from node groups — matches the
// physics-side frame() in car.rs. Used to express the skinning offsets in a
// frame that rotates with the car, so the body doesn't shear when it turns.
function chassisFrame(pos, g) {
  const fc = centroidOf(pos, g.front), rc = centroidOf(pos, g.rear);
  const rightc = centroidOf(pos, g.right), leftc = centroidOf(pos, g.left);
  const fwd = norm(sub(fc, rc));
  const up = norm(cross(sub(rightc, leftc), fwd));
  const right = norm(cross(fwd, up));
  return [fwd, right, up];
}

// Build the skinnable car body. `carDesc.chassisRest` is the flat [x,y,z,...] of
// the chassis nodes' world rest positions; `chassisCount` is how many.
export function buildCarBody(carDesc) {
  const { cageMin: min, cageMax: max, bodyColor: color } = carDesc;
  const rest = carDesc.chassisRest; // flat, world space
  const nNodes = carDesc.chassisCount;

  const hull = carHull(min, max);
  const vCount = hull.positions.length;
  const tris = hull.tris;

  // Author space -> world: the hull is in local cage coords, the chassis rest is
  // in world. They differ by the spawn translation; recover it from centroids.
  let rcx = 0, rcy = 0, rcz = 0;
  for (let c = 0; c < nNodes; c++) {
    rcx += rest[c * 3];
    rcy += rest[c * 3 + 1];
    rcz += rest[c * 3 + 2];
  }
  rcx /= nNodes; rcy /= nNodes; rcz /= nNodes;
  const lcx = (min[0] + max[0]) / 2, lcy = (min[1] + max[1]) / 2, lcz = (min[2] + max[2]) / 2;
  const base = [rcx - lcx, rcy - lcy, rcz - lcz]; // local -> world translation

  // Identify the extreme-face node groups (front=+x, rear=-x, right=+z, left=-z),
  // the same groups the physics uses to define the car's frame. The skinning
  // offsets are stored in this frame so they rotate with the car (no shear when
  // turning) while local node motion still crumples the panels.
  let xmin = Infinity, xmax = -Infinity, zmin = Infinity, zmax = -Infinity;
  for (let c = 0; c < nNodes; c++) {
    xmin = Math.min(xmin, rest[c * 3]); xmax = Math.max(xmax, rest[c * 3]);
    zmin = Math.min(zmin, rest[c * 3 + 2]); zmax = Math.max(zmax, rest[c * 3 + 2]);
  }
  const ex = (xmax - xmin) * 0.01 + 1e-4, ez = (zmax - zmin) * 0.01 + 1e-4;
  const groups = { front: [], rear: [], left: [], right: [] };
  for (let c = 0; c < nNodes; c++) {
    if (rest[c * 3] >= xmax - ex) groups.front.push(c);
    if (rest[c * 3] <= xmin + ex) groups.rear.push(c);
    if (rest[c * 3 + 2] >= zmax - ez) groups.right.push(c);
    if (rest[c * 3 + 2] <= zmin + ez) groups.left.push(c);
  }
  const restBasis = chassisFrame(rest, groups); // [fwd, right, up] at rest

  // For each vertex: K nearest chassis nodes, inverse-square weights, and a
  // local-frame offset (stored in restBasis coords).
  const indices = new Uint16Array(vCount * K);
  const weights = new Float32Array(vCount * K);
  const offset = new Float32Array(vCount * 3);

  for (let v = 0; v < vCount; v++) {
    const wx = hull.positions[v][0] + base[0];
    const wy = hull.positions[v][1] + base[1];
    const wz = hull.positions[v][2] + base[2];

    // Find K nearest nodes (small nNodes -> simple partial selection).
    const best = []; // {d2, idx}
    for (let c = 0; c < nNodes; c++) {
      const dx = rest[c * 3] - wx, dy = rest[c * 3 + 1] - wy, dz = rest[c * 3 + 2] - wz;
      const d2 = dx * dx + dy * dy + dz * dz;
      best.push({ d2, idx: c });
    }
    best.sort((a, b) => a.d2 - b.d2);

    let wsum = 0;
    const ws = [];
    for (let k = 0; k < K; k++) {
      const w = 1 / (best[k].d2 + 1e-4); // inverse square distance
      ws.push(w);
      wsum += w;
    }
    let px = 0, py = 0, pz = 0;
    for (let k = 0; k < K; k++) {
      const w = ws[k] / wsum;
      const idx = best[k].idx;
      indices[v * K + k] = idx;
      weights[v * K + k] = w;
      px += w * rest[idx * 3];
      py += w * rest[idx * 3 + 1];
      pz += w * rest[idx * 3 + 2];
    }
    // World offset preserves the exact rest shape, then projected into the car's
    // rest frame so it can be rotated back with the live frame during skinning.
    const ow = [wx - px, wy - py, wz - pz];
    offset[v * 3] = dot(ow, restBasis[0]);
    offset[v * 3 + 1] = dot(ow, restBasis[1]);
    offset[v * 3 + 2] = dot(ow, restBasis[2]);
  }

  const triIndices = new Uint16Array(tris.length * 3);
  for (let t = 0; t < tris.length; t++) {
    triIndices[t * 3] = tris[t][0];
    triIndices[t * 3 + 1] = tris[t][1];
    triIndices[t * 3 + 2] = tris[t][2];
  }

  const interleaved = new Float32Array(vCount * 6);

  // Rebuild world vertices from current chassis node positions (flat [x,y,z,...]).
  function skin(chassisPos) {
    // Live car frame, so the baked offsets rotate with the car instead of
    // shearing the mesh when it turns.
    const [fwd, right, up] = chassisFrame(chassisPos, groups);
    for (let v = 0; v < vCount; v++) {
      let x = 0, y = 0, z = 0;
      for (let k = 0; k < K; k++) {
        const idx = indices[v * K + k];
        const w = weights[v * K + k];
        x += w * chassisPos[idx * 3];
        y += w * chassisPos[idx * 3 + 1];
        z += w * chassisPos[idx * 3 + 2];
      }
      // Rotate the local-frame offset into world space: ol.x*fwd + ol.y*right + ol.z*up.
      const ox = offset[v * 3], oy = offset[v * 3 + 1], oz = offset[v * 3 + 2];
      const o = v * 6;
      interleaved[o] = x + ox * fwd[0] + oy * right[0] + oz * up[0];
      interleaved[o + 1] = y + ox * fwd[1] + oy * right[1] + oz * up[1];
      interleaved[o + 2] = z + ox * fwd[2] + oy * right[2] + oz * up[2];
      interleaved[o + 3] = 0;
      interleaved[o + 4] = 0;
      interleaved[o + 5] = 0;
    }
    for (let t = 0; t < triIndices.length; t += 3) {
      const a = triIndices[t] * 6, b = triIndices[t + 1] * 6, c = triIndices[t + 2] * 6;
      const e1x = interleaved[b] - interleaved[a];
      const e1y = interleaved[b + 1] - interleaved[a + 1];
      const e1z = interleaved[b + 2] - interleaved[a + 2];
      const e2x = interleaved[c] - interleaved[a];
      const e2y = interleaved[c + 1] - interleaved[a + 1];
      const e2z = interleaved[c + 2] - interleaved[a + 2];
      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;
      interleaved[a + 3] += nx; interleaved[a + 4] += ny; interleaved[a + 5] += nz;
      interleaved[b + 3] += nx; interleaved[b + 4] += ny; interleaved[b + 5] += nz;
      interleaved[c + 3] += nx; interleaved[c + 4] += ny; interleaved[c + 5] += nz;
    }
    for (let v = 0; v < vCount; v++) {
      const o = v * 6 + 3;
      const len = Math.hypot(interleaved[o], interleaved[o + 1], interleaved[o + 2]) || 1;
      interleaved[o] /= len;
      interleaved[o + 1] /= len;
      interleaved[o + 2] /= len;
    }
    return interleaved;
  }

  return { vCount, triIndices, color, interleaved, skin };
}
