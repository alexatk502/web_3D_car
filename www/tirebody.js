// Deformable tire render mesh. The physics tread ring is a single mid-plane ring
// of N nodes around the hub; this extrudes it to a width-having wheel (a cylinder
// band + two end caps) by offsetting each tread node ±halfWidth along the ring's
// axle (its plane normal). Re-skinned every frame from the live node positions,
// so the rendered wheel squats/deforms/blows out with the physics.

function cross(ax, ay, az, bx, by, bz) {
  return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
}
function norm3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

// `treadN` tread nodes, `halfWidth` extrusion, `color`. Vertex layout:
//   [left_0..left_{N-1}][right_0..right_{N-1}][hub_left][hub_right]  → 2N+2 verts.
export function buildTire(treadN, halfWidth, color) {
  const N = treadN;
  const vCount = 2 * N + 2;
  const HUB_L = 2 * N;
  const HUB_R = 2 * N + 1;
  const tris = [];
  for (let i = 0; i < N; i++) {
    const j = (i + 1) % N;
    const li = i, lj = j, ri = N + i, rj = N + j;
    // Outer tread band.
    tris.push([li, lj, ri], [ri, lj, rj]);
    // Side caps (fans from the two hub centres).
    tris.push([HUB_L, lj, li]);
    tris.push([HUB_R, ri, rj]);
  }
  const triIndices = new Uint16Array(tris.length * 3);
  for (let t = 0; t < tris.length; t++) {
    triIndices[t * 3] = tris[t][0];
    triIndices[t * 3 + 1] = tris[t][1];
    triIndices[t * 3 + 2] = tris[t][2];
  }

  const interleaved = new Float32Array(vCount * 6);

  // hub = [x,y,z]; tread = flat [x,y,z,...] of the N tread nodes (world space).
  function skin(hub, tread) {
    // Axle = normal of the ring plane, from two spokes a quarter-turn apart.
    const q = (N >> 2) || 1;
    const e0 = [tread[0] - hub[0], tread[1] - hub[1], tread[2] - hub[2]];
    const e1 = [tread[q * 3] - hub[0], tread[q * 3 + 1] - hub[1], tread[q * 3 + 2] - hub[2]];
    const n = norm3(cross(e0[0], e0[1], e0[2], e1[0], e1[1], e1[2]));
    const hw = halfWidth;

    const setV = (idx, x, y, z) => {
      const o = idx * 6;
      interleaved[o] = x;
      interleaved[o + 1] = y;
      interleaved[o + 2] = z;
      interleaved[o + 3] = 0;
      interleaved[o + 4] = 0;
      interleaved[o + 5] = 0;
    };
    for (let i = 0; i < N; i++) {
      const tx = tread[i * 3], ty = tread[i * 3 + 1], tz = tread[i * 3 + 2];
      setV(i, tx + n[0] * hw, ty + n[1] * hw, tz + n[2] * hw); // left
      setV(N + i, tx - n[0] * hw, ty - n[1] * hw, tz - n[2] * hw); // right
    }
    setV(HUB_L, hub[0] + n[0] * hw, hub[1] + n[1] * hw, hub[2] + n[2] * hw);
    setV(HUB_R, hub[0] - n[0] * hw, hub[1] - n[1] * hw, hub[2] - n[2] * hw);

    // Recompute vertex normals from face normals (same as carbody.js).
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
      const l = Math.hypot(interleaved[o], interleaved[o + 1], interleaved[o + 2]) || 1;
      interleaved[o] /= l;
      interleaved[o + 1] /= l;
      interleaved[o + 2] /= l;
    }
    return interleaved;
  }

  return { vCount, triIndices, color, interleaved, skin };
}
