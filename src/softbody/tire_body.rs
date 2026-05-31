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

const PRESSURE: f32 = 700.0; // radial outward force per tread node when inflated
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
) -> Vec<u32> {
    let c = Vec3::from_array(center);
    let (av, bv) = (Vec3::from_array(a), Vec3::from_array(b));
    let mut ids = Vec::with_capacity(TREAD_N);
    for i in 0..TREAD_N {
        let theta = (i as f32) * std::f32::consts::TAU / TREAD_N as f32;
        let p = c + (av * theta.cos() + bv * theta.sin()) * r;
        let id = nodes.push(p.to_array(), TREAD_MASS, TREAD_RADIUS);
        nodes.mark_tire(id);
        ids.push(id);
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
    ids
}

fn push_beam(nodes: &Nodes, beams: &mut Beams, a: u32, b: u32, k: f32, d: f32) {
    let (ia, ib) = (a as usize, b as usize);
    let dx = nodes.px[ia] - nodes.px[ib];
    let dy = nodes.py[ia] - nodes.py[ib];
    let dz = nodes.pz[ia] - nodes.pz[ib];
    let rest = (dx * dx + dy * dy + dz * dz).sqrt();
    beams.push(a, b, rest, k, d, TIRE_DEFORM, TIRE_BREAK, BeamKind::Tire);
}

/// Per-substep tire forces on one wheel's tread ring: inflation pressure, an
/// axle-centering spring (keeps the ring planar), and ground squat contact. The
/// hub frame is `(a, b)` (wheel plane) + `axle` (spin axis). `pressure` is 0..1
/// (0 = blown flat). Spoke/ring beam forces come from the generic solver; this
/// adds only the tire-specific extras.
///
/// NOTE: the ring is deliberately NOT spun physically. A mass on a spring
/// orbiting in a circle gains radius every step under explicit integration (an
/// orbit blow-up), so a tangential spin motor made the tires grow without bound
/// while driving. The ring only deforms radially here; the rolling rotation is
/// invisible on a plain tire and the hub still carries `omega` for the physics.
pub fn apply_tire(nodes: &mut Nodes, tread: &[u32], hub: usize, a: Vec3, b: Vec3, axle: Vec3, pressure: f32) {
    let hub_p = Vec3::new(nodes.px[hub], nodes.py[hub], nodes.pz[hub]);
    for &t in tread {
        let i = t as usize;
        let p = Vec3::new(nodes.px[i], nodes.py[i], nodes.pz[i]);
        let v = Vec3::new(nodes.vx[i], nodes.vy[i], nodes.vz[i]);
        let rel = p - hub_p;

        // In-plane radial direction (wheel plane spanned by a, b).
        let ca = rel.dot(a);
        let cb = rel.dot(b);
        let radius_ip = (ca * ca + cb * cb).sqrt().max(1e-4);
        let outward = (a * ca + b * cb) / radius_ip;

        // Inflation pressure: push the tread radially outward (shape + springiness).
        let mut f = outward * (PRESSURE * pressure);

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

        nodes.fx[i] += f.x;
        nodes.fy[i] += f.y;
        nodes.fz[i] += f.z;
    }
}
