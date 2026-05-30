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

  // For each vertex: K nearest chassis nodes, inverse-square weights, rest offset.
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
    // Offset preserves the exact rest shape (vertex = predicted + offset).
    offset[v * 3] = wx - px;
    offset[v * 3 + 1] = wy - py;
    offset[v * 3 + 2] = wz - pz;
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
    for (let v = 0; v < vCount; v++) {
      let x = 0, y = 0, z = 0;
      for (let k = 0; k < K; k++) {
        const idx = indices[v * K + k];
        const w = weights[v * K + k];
        x += w * chassisPos[idx * 3];
        y += w * chassisPos[idx * 3 + 1];
        z += w * chassisPos[idx * 3 + 2];
      }
      const o = v * 6;
      interleaved[o] = x + offset[v * 3];
      interleaved[o + 1] = y + offset[v * 3 + 1];
      interleaved[o + 2] = z + offset[v * 3 + 2];
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
