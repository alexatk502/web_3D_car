//! Deformable node/beam tire — a ring of tread nodes around a wheel hub.
//!
//! The hub keeps the analytic grip/support model (proven stable); this adds the
//! *visible, deformable shell*: a tread ring held to shape by spoke (hub→tread)
//! and ring (tread→tread) beams plus an inflation pressure force pushing the tread
//! radially out. It squats where the ground/obstacles push a tread node inward,
//! deforms over bumps, and goes flat when its beams break or pressure is lost
//! (blowout). The ring is spun by a tangential "motor" that tracks the wheel's
//! scalar `omega`, so the visual rotation matches the drivetrain without making
//! `omega` itself emerge from noisy node motion.

use crate::scene;
use crate::softbody::structure::{BeamKind, Beams, Nodes};
use glam::Vec3;

pub const TREAD_N: usize = 12; // tread nodes per tire
const TREAD_MASS: f32 = 1.2;
const TREAD_RADIUS: f32 = 0.07; // node collision radius (vs obstacles)

const SPOKE_K: f32 = 26_000.0; // hub→tread (radial); pressure does most shape-holding
const SPOKE_D: f32 = 600.0;
const RING_K: f32 = 60_000.0; // tread→tread (circumferential)
const RING_D: f32 = 600.0;
const TIRE_DEFORM: f32 = 0.18; // flexes a lot before taking a set
const TIRE_BREAK: f32 = 0.55; // ring/spoke severs past this strain → blowout

const PRESSURE: f32 = 2_000.0; // radial outward force per tread node when inflated
const MOTOR_K: f32 = 5_000.0; // angular spring driving the ring to spin with omega
const MOTOR_D: f32 = 80.0; // tangential velocity damping
const AXLE_K: f32 = 9_000.0; // keeps the ring planar (resists sideways splay)

const TREAD_CONTACT_K: f32 = 34_000.0; // squat stiffness vs ground
const TREAD_CONTACT_D: f32 = 500.0;

/// Build a tread ring of `TREAD_N` nodes around `hub` (at local `center`), radius
/// `r`, in the wheel plane spanned by unit vectors `a`, `b`. Adds spoke and ring
/// beams (BeamKind::Tire). Returns (tread node ids, their rest angles). Built in
/// the car's pre-transform local space, so the whole-car spawn transform applies.
pub fn build_tire(
    nodes: &mut Nodes,
    beams: &mut Beams,
    hub: u32,
    center: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    r: f32,
) -> (Vec<u32>, Vec<f32>) {
    let c = Vec3::from_array(center);
    let (av, bv) = (Vec3::from_array(a), Vec3::from_array(b));
    let mut ids = Vec::with_capacity(TREAD_N);
    let mut angles = Vec::with_capacity(TREAD_N);
    for i in 0..TREAD_N {
        let theta = (i as f32) * std::f32::consts::TAU / TREAD_N as f32;
        let p = c + (av * theta.cos() + bv * theta.sin()) * r;
        let id = nodes.push(p.to_array(), TREAD_MASS, TREAD_RADIUS);
        nodes.mark_tire(id);
        ids.push(id);
        angles.push(theta);
    }
    // Spokes: hub → each tread node.
    for &t in &ids {
        push_beam(nodes, beams, hub, t, SPOKE_K, SPOKE_D);
    }
    // Ring: tread[i] → tread[i+1].
    for i in 0..TREAD_N {
        let t0 = ids[i];
        let t1 = ids[(i + 1) % TREAD_N];
        push_beam(nodes, beams, t0, t1, RING_K, RING_D);
    }
    (ids, angles)
}

fn push_beam(nodes: &Nodes, beams: &mut Beams, a: u32, b: u32, k: f32, d: f32) {
    let (ia, ib) = (a as usize, b as usize);
    let dx = nodes.px[ia] - nodes.px[ib];
    let dy = nodes.py[ia] - nodes.py[ib];
    let dz = nodes.pz[ia] - nodes.pz[ib];
    let rest = (dx * dx + dy * dy + dz * dz).sqrt();
    beams.push(a, b, rest, k, d, TIRE_DEFORM, TIRE_BREAK, BeamKind::Tire);
}

/// Per-substep tire forces on one wheel's tread ring: inflation pressure, the
/// spin motor (tracks `spin`), an axle-centering spring (keeps the ring planar),
/// and ground squat contact. The hub frame is `(a, b)` (wheel plane) + `axle`
/// (spin axis). `pressure` is 0..1 (0 = blown flat). Spoke/ring beam forces come
/// from the generic solver; this only adds the tire-specific extras.
#[allow(clippy::too_many_arguments)]
pub fn apply_tire(
    nodes: &mut Nodes,
    tread: &[u32],
    angles: &[f32],
    hub: usize,
    a: Vec3,
    b: Vec3,
    axle: Vec3,
    r: f32,
    spin: f32,
    pressure: f32,
    dt: f32,
) {
    let hub_p = Vec3::new(nodes.px[hub], nodes.py[hub], nodes.pz[hub]);
    let _ = dt;
    for (k, &t) in tread.iter().enumerate() {
        let i = t as usize;
        let p = Vec3::new(nodes.px[i], nodes.py[i], nodes.pz[i]);
        let v = Vec3::new(nodes.vx[i], nodes.vy[i], nodes.vz[i]);
        let rel = p - hub_p;

        // In-plane (wheel-plane) decomposition.
        let ca = rel.dot(a);
        let cb = rel.dot(b);
        let radius_ip = (ca * ca + cb * cb).sqrt().max(1e-4);
        let outward = (a * ca + b * cb) / radius_ip; // unit radial (in-plane)
        let tang = (a * (-cb) + b * ca) / radius_ip; // unit tangential (CCW)

        let mut f = Vec3::ZERO;

        // Inflation pressure: push the tread radially outward (shape + springiness).
        f += outward * (PRESSURE * pressure);

        // Spin motor: drive the node toward its target angle (rest_angle + spin),
        // damped on tangential velocity. Leaves the radial direction free (squat).
        let target = angles[k] + spin;
        let current = cb.atan2(ca);
        let mut err = target - current;
        // Wrap to (-π, π].
        err = err - std::f32::consts::TAU * (err / std::f32::consts::TAU).round();
        let tang_vel = v.dot(tang);
        f += tang * (MOTOR_K * err * radius_ip - MOTOR_D * tang_vel);

        // Axle-centering: keep the ring in its plane (resist sideways splay).
        let axle_comp = rel.dot(axle);
        f -= axle * (AXLE_K * axle_comp);

        // Ground squat: normal-only penalty (no friction — the hub tire owns grip).
        let ground = scene::terrain_height(p.x, p.z);
        let pen = (ground + nodes.radius[i]) - p.y;
        if pen > 0.0 {
            let nn = scene::terrain_normal(p.x, p.z);
            let nrm = Vec3::new(nn[0], nn[1], nn[2]);
            let vn = v.dot(nrm);
            let fz = (TREAD_CONTACT_K * pen - TREAD_CONTACT_D * vn).max(0.0);
            f += nrm * fz;
        }
        let _ = r;

        nodes.fx[i] += f.x;
        nodes.fy[i] += f.y;
        nodes.fz[i] += f.z;
    }
}
