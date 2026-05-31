//! Analytic collision against the procedural terrain. Because the terrain is a
//! closed-form height field (`scene::terrain_height`), node collision is a cheap
//! per-node sample with no broadphase — ideal for the high substep rate.

use crate::scene::{self, ObstacleBox};
use crate::softbody::structure::Nodes;
use rayon::prelude::*;

// Penalty-contact tuning.
const CONTACT_K: f32 = 120_000.0; // ground stiffness (N/m of penetration)
const CONTACT_DAMP: f32 = 600.0; // normal velocity damping
const FRICTION: f32 = 0.9; // Coulomb-ish tangential coefficient

// Obstacle (box) + self-collision tuning.
const OBSTACLE_K: f32 = 160_000.0;
const OBSTACLE_DAMP: f32 = 800.0;
const SELF_K: f32 = 60_000.0; // node-node repulsion stiffness

// Vehicle-vehicle contact. Damped (unlike self-collision) because cars meet at
// speed and an undamped penalty would explode on deep one-step interpenetration.
const CROSS_K: f32 = 120_000.0;
const CROSS_D: f32 = 3_000.0;

/// Add ground contact forces to any node penetrating the terrain. Uses the
/// terrain surface normal (gradient) for both push-out and friction.
pub fn apply_terrain(n: &mut Nodes) {
    for i in 0..n.len() {
        if n.inv_mass[i] == 0.0 || n.is_wheel[i] {
            continue; // pinned, or a wheel hub (handled by the tire model)
        }
        let r = n.radius[i];
        let ground = scene::terrain_height(n.px[i], n.pz[i]);
        let penetration = (ground + r) - n.py[i];
        if penetration <= 0.0 {
            continue;
        }

        let [nx, ny, nz] = scene::terrain_normal(n.px[i], n.pz[i]);

        // Normal velocity component.
        let vn = n.vx[i] * nx + n.vy[i] * ny + n.vz[i] * nz;

        // Penalty normal force: spring out of the surface, damped on approach.
        let mut fn_mag = CONTACT_K * penetration - CONTACT_DAMP * vn;
        if fn_mag < 0.0 {
            fn_mag = 0.0; // contacts only push, never pull
        }
        n.fx[i] += fn_mag * nx;
        n.fy[i] += fn_mag * ny;
        n.fz[i] += fn_mag * nz;

        // Coulomb friction opposing the tangential velocity.
        let vtx = n.vx[i] - vn * nx;
        let vty = n.vy[i] - vn * ny;
        let vtz = n.vz[i] - vn * nz;
        let vt = (vtx * vtx + vty * vty + vtz * vtz).sqrt();
        if vt > 1e-4 {
            let max_friction = FRICTION * scene::surface_friction(n.px[i], n.pz[i]) * fn_mag;
            // Force opposes motion; capped by the friction cone.
            let scale = -max_friction / vt;
            n.fx[i] += vtx * scale;
            n.fy[i] += vty * scale;
            n.fz[i] += vtz * scale;
        }
    }
}

/// Penalty collision of every (non-pinned) node against the axis-aligned obstacle
/// boxes. Uses the classic sphere-vs-AABB closest-point test. This is what makes
/// crashing into the boxes deform the car.
pub fn apply_obstacles(n: &mut Nodes, boxes: &[ObstacleBox]) {
    for i in 0..n.len() {
        if n.inv_mass[i] == 0.0 {
            continue;
        }
        let (px, py, pz) = (n.px[i], n.py[i], n.pz[i]);
        let r = n.radius[i];
        for b in boxes {
            let (min, max) = (
                [b.center[0] - b.half[0], b.center[1] - b.half[1], b.center[2] - b.half[2]],
                [b.center[0] + b.half[0], b.center[1] + b.half[1], b.center[2] + b.half[2]],
            );
            // Closest point on the box to the node center.
            let cx = px.clamp(min[0], max[0]);
            let cy = py.clamp(min[1], max[1]);
            let cz = pz.clamp(min[2], max[2]);
            let (dx, dy, dz) = (px - cx, py - cy, pz - cz);
            let dist2 = dx * dx + dy * dy + dz * dz;

            let (nx, ny, nz, pen);
            if dist2 > 1e-12 {
                // Center outside the box: contact only if the surface is within r.
                if dist2 >= r * r {
                    continue;
                }
                let dist = dist2.sqrt();
                nx = dx / dist;
                ny = dy / dist;
                nz = dz / dist;
                pen = r - dist;
            } else {
                // Center inside the box: eject along the least-penetration face,
                // by the full depth to that face PLUS the radius (otherwise a
                // deeply embedded node only gets a tiny push and stays lodged).
                let dl = [px - min[0], py - min[1], pz - min[2]];
                let dh = [max[0] - px, max[1] - py, max[2] - pz];
                let axes = [
                    (dl[0], [-1.0, 0.0, 0.0]),
                    (dh[0], [1.0, 0.0, 0.0]),
                    (dl[1], [0.0, -1.0, 0.0]),
                    (dh[1], [0.0, 1.0, 0.0]),
                    (dl[2], [0.0, 0.0, -1.0]),
                    (dh[2], [0.0, 0.0, 1.0]),
                ];
                let mut best = f32::INFINITY;
                let mut axis = [0.0, 1.0, 0.0];
                for (d, a) in axes {
                    if d < best {
                        best = d;
                        axis = a;
                    }
                }
                nx = axis[0];
                ny = axis[1];
                nz = axis[2];
                pen = best + r;
            }

            let vn = n.vx[i] * nx + n.vy[i] * ny + n.vz[i] * nz;
            let f = (OBSTACLE_K * pen - OBSTACLE_DAMP * vn).max(0.0);
            n.fx[i] += f * nx;
            n.fy[i] += f * ny;
            n.fz[i] += f * nz;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::softbody::structure::Nodes;

    /// A node spawned deep inside a box must be ejected (not lodged) — regression
    /// for the "fall onto a block and get stuck" bug.
    #[test]
    fn node_inside_box_is_ejected() {
        let mut n = Nodes::default();
        n.push([0.0, 1.0, 0.0], 5.0, 0.2); // at the box center
        let boxes = vec![ObstacleBox {
            center: [0.0, 1.0, 0.0],
            half: [1.0, 1.0, 1.0],
            color: [0.0, 0.0, 0.0],
        }];

        // Penalty contact + plain Euler, no gravity.
        let dt = 1.0 / 1000.0;
        for _ in 0..400 {
            n.fx[0] = 0.0;
            n.fy[0] = 0.0;
            n.fz[0] = 0.0;
            apply_obstacles(&mut n, &boxes);
            let im = n.inv_mass[0];
            n.vx[0] += n.fx[0] * im * dt;
            n.vy[0] += n.fy[0] * im * dt;
            n.vz[0] += n.fz[0] * im * dt;
            n.px[0] += n.vx[0] * dt;
            n.py[0] += n.vy[0] * dt;
            n.pz[0] += n.vz[0] * dt;
        }

        // Outside the box (beyond a face by at least the radius margin).
        let outside = n.px[0].abs() > 1.0 || n.py[0] > 2.0 || n.py[0] < 0.0 || n.pz[0].abs() > 1.0;
        assert!(outside, "node should be ejected, ended at ({}, {}, {})", n.px[0], n.py[0], n.pz[0]);
    }

    // A tight grid of overlapping nodes (none beam-connected). The parallel
    // gather must produce the same net per-node force as the serial scatter.
    fn overlapping_grid(d: usize) -> Nodes {
        let mut n = Nodes::default();
        for x in 0..d {
            for y in 0..d {
                for z in 0..d {
                    // Spacing 0.3 < radius-sum 0.4 -> neighbours overlap.
                    n.push([x as f32 * 0.3, y as f32 * 0.3, z as f32 * 0.3], 1.0, 0.2);
                }
            }
        }
        n
    }

    #[test]
    fn parallel_self_collision_matches_serial() {
        let mut n = overlapping_grid(5); // 125 nodes
        let never = |_: usize, _: usize| false;

        // Serial scatter into the force accumulators.
        for i in 0..n.len() {
            n.fx[i] = 0.0; n.fy[i] = 0.0; n.fz[i] = 0.0;
        }
        apply_self_collision(&mut n, never);
        let serial: Vec<[f32; 3]> = (0..n.len()).map(|i| [n.fx[i], n.fy[i], n.fz[i]]).collect();

        // Parallel gather.
        let mut par = vec![[0.0f32; 3]; n.len()];
        let mut bufs = Vec::new();
        self_collision_gather(&n, never, &mut bufs, &mut par);

        for i in 0..n.len() {
            for c in 0..3 {
                let diff = (serial[i][c] - par[i][c]).abs();
                assert!(
                    diff < 1e-2,
                    "node {} comp {}: serial {} vs parallel {}",
                    i, c, serial[i][c], par[i][c]
                );
            }
        }
    }

    #[test]
    fn cross_body_collision_pushes_apart_and_conserves_momentum() {
        // Two single-node "cars" overlapping (centres 0.2 apart, radii 0.2 → rsum 0.4).
        let mut a = Nodes::default();
        a.push([0.0, 0.0, 0.0], 1.0, 0.2);
        let mut b = Nodes::default();
        b.push([0.2, 0.0, 0.0], 1.0, 0.2);

        cross_body_collision(&mut a, &mut b);

        // a is pushed -x, b is pushed +x (apart along the centre line).
        assert!(a.fx[0] < 0.0, "a should be pushed -x (got {})", a.fx[0]);
        assert!(b.fx[0] > 0.0, "b should be pushed +x (got {})", b.fx[0]);
        // Equal and opposite → total force ~0 (momentum conserving).
        assert!((a.fx[0] + b.fx[0]).abs() < 1e-3, "forces should cancel");
    }

    // Scaling check — run explicitly:
    //   cargo test --release bench_self_collision -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_self_collision_scaling() {
        use std::time::Instant;
        let mut n = overlapping_grid(12); // 1728 nodes
        let never = |_: usize, _: usize| false;
        let iters = 40;

        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..n.len() {
                n.fx[i] = 0.0; n.fy[i] = 0.0; n.fz[i] = 0.0;
            }
            apply_self_collision(&mut n, never);
        }
        let serial = t0.elapsed();

        let mut out = vec![[0.0f32; 3]; n.len()];
        let mut bufs = Vec::new();
        let t1 = Instant::now();
        for _ in 0..iters {
            self_collision_gather(&n, never, &mut bufs, &mut out);
        }
        let parallel = t1.elapsed();

        println!(
            "self-collision {} nodes x{} iters: serial {:?}, parallel {:?}, speedup {:.2}x ({} threads)",
            n.len(), iters, serial, parallel,
            serial.as_secs_f64() / parallel.as_secs_f64(),
            rayon::current_num_threads()
        );
    }
}

/// Parallel, race-free self-collision returning the net force per node in `out`.
/// Spawns exactly `num_threads` tasks, each handling a *strided* subset of the
/// outer index `i` (t, t+T, t+2T, …). Striding balances the triangular workload
/// (low and high `i` interleaved) and means just T tasks with no work-stealing
/// churn. Each task accumulates the symmetric pair forces (both endpoints, j>i)
/// into its own buffer in `bufs` (reused across substeps — no per-call alloc),
/// and the T buffers are summed into `out`. Total pair work is n^2/2, same as
/// the serial scatter. `connected(i,j)` must be order-independent and `Sync`.
pub fn self_collision_gather<F>(
    n: &Nodes,
    connected: F,
    bufs: &mut Vec<Vec<[f32; 3]>>,
    out: &mut [[f32; 3]],
) where
    F: Fn(usize, usize) -> bool + Sync,
{
    let count = n.len();
    let nt = rayon::current_num_threads().max(1);
    bufs.resize_with(nt, Vec::new);
    for b in bufs.iter_mut() {
        b.clear();
        b.resize(count, [0.0; 3]);
    }

    bufs.par_iter_mut().enumerate().for_each(|(t, buf)| {
        let mut i = t;
        while i < count {
            if n.inv_mass[i] != 0.0 {
                let (pxi, pyi, pzi, ri) = (n.px[i], n.py[i], n.pz[i], n.radius[i]);
                for j in (i + 1)..count {
                    if n.inv_mass[j] == 0.0 || connected(i, j) {
                        continue;
                    }
                    let dx = n.px[j] - pxi;
                    let dy = n.py[j] - pyi;
                    let dz = n.pz[j] - pzi;
                    let rsum = ri + n.radius[j];
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    if dist2 >= rsum * rsum || dist2 < 1e-9 {
                        continue;
                    }
                    let dist = dist2.sqrt();
                    let f = SELF_K * (rsum - dist) / dist;
                    let (fx, fy, fz) = (f * dx, f * dy, f * dz);
                    // Push apart: j gets +, i gets − (same convention as serial).
                    buf[j][0] += fx; buf[j][1] += fy; buf[j][2] += fz;
                    buf[i][0] -= fx; buf[i][1] -= fy; buf[i][2] -= fz;
                }
            }
            i += nt;
        }
    });

    for f in out.iter_mut() {
        *f = [0.0; 3];
    }
    for buf in bufs.iter() {
        for k in 0..count {
            out[k][0] += buf[k][0];
            out[k][1] += buf[k][1];
            out[k][2] += buf[k][2];
        }
    }
}

/// Vehicle-vehicle collision: bipartite sphere-sphere damped penalty between the
/// nodes of two different cars. Because the bodies are never beam-connected there
/// is no skip set — every overlapping cross-pair repels. Writes directly into both
/// force accumulators (the two `Nodes` are disjoint, so `&mut` to each is safe).
/// Momentum-conserving: equal and opposite force on each pair.
pub fn cross_body_collision(a: &mut Nodes, b: &mut Nodes) {
    for i in 0..a.len() {
        if a.inv_mass[i] == 0.0 {
            continue;
        }
        let (pxi, pyi, pzi, ri) = (a.px[i], a.py[i], a.pz[i], a.radius[i]);
        let (vxi, vyi, vzi) = (a.vx[i], a.vy[i], a.vz[i]);
        for j in 0..b.len() {
            if b.inv_mass[j] == 0.0 {
                continue;
            }
            let dx = b.px[j] - pxi;
            let dy = b.py[j] - pyi;
            let dz = b.pz[j] - pzi;
            let rsum = ri + b.radius[j];
            let dist2 = dx * dx + dy * dy + dz * dz;
            if dist2 >= rsum * rsum || dist2 < 1e-9 {
                continue;
            }
            let dist = dist2.sqrt();
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let pen = rsum - dist;
            // Relative normal velocity (b approaching a is negative → adds push).
            let rvn = (b.vx[j] - vxi) * ux + (b.vy[j] - vyi) * uy + (b.vz[j] - vzi) * uz;
            let f = (CROSS_K * pen - CROSS_D * rvn).max(0.0); // only push, never pull
            b.fx[j] += f * ux;
            b.fy[j] += f * uy;
            b.fz[j] += f * uz;
            a.fx[i] -= f * ux;
            a.fy[i] -= f * uy;
            a.fz[i] -= f * uz;
        }
    }
}

/// Light brute-force self-collision: repel node pairs that are NOT joined by an
/// (unbroken) beam and have come closer than the sum of their radii. Keeps a
/// crumpled structure from passing through itself. O(n^2) — fine for a car-sized
/// node count. `connected(i,j)` reports an unbroken beam between i and j.
pub fn apply_self_collision<F: Fn(usize, usize) -> bool>(n: &mut Nodes, connected: F) {
    let count = n.len();
    for i in 0..count {
        if n.inv_mass[i] == 0.0 {
            continue;
        }
        for j in (i + 1)..count {
            if n.inv_mass[j] == 0.0 || connected(i, j) {
                continue;
            }
            let dx = n.px[j] - n.px[i];
            let dy = n.py[j] - n.py[i];
            let dz = n.pz[j] - n.pz[i];
            let rsum = n.radius[i] + n.radius[j];
            let dist2 = dx * dx + dy * dy + dz * dz;
            if dist2 >= rsum * rsum || dist2 < 1e-9 {
                continue;
            }
            let dist = dist2.sqrt();
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let pen = rsum - dist;
            let f = SELF_K * pen;
            // Push apart (j gets +, i gets −).
            n.fx[j] += f * ux;
            n.fy[j] += f * uy;
            n.fz[j] += f * uz;
            n.fx[i] -= f * ux;
            n.fy[i] -= f * uy;
            n.fz[i] -= f * uz;
        }
    }
}
