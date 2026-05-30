//! Explicit soft-body integrator. Semi-implicit (symplectic) Euler at a high
//! fixed substep rate — the same family of solver BeamNG uses. Stiff beams stay
//! stable because the substep dt is tiny and beams are damped.

use crate::softbody::collision;
use crate::softbody::structure::Structure;

/// Solver tuning. `substep_dt` is the high-rate physics step; the render loop
/// runs many substeps per frame.
pub struct SolverParams {
    pub gravity: f32,
    pub substep_dt: f32,
    /// Global velocity damping coefficient **per second** (air drag-ish). Scaled
    /// by dt each substep so it is frame-rate independent and stays light.
    pub global_damping: f32,
    /// Fraction of the beyond-yield deviation absorbed as permanent set each
    /// substep (plastic creep rate). 0 = perfectly elastic.
    pub plastic_rate: f32,
}

impl Default for SolverParams {
    fn default() -> Self {
        SolverParams {
            gravity: -20.0,
            substep_dt: 1.0 / 1000.0, // 1 kHz in Phase 0 (single-thread); raised later
            global_damping: 0.25, // ~light air drag; lets the body wobble
            plastic_rate: 0.4,
        }
    }
}

/// Advance a plain structure (no vehicle) by one substep: beam forces, terrain
/// collision, integrate. Used for generic bodies and tests.
pub fn substep(s: &mut Structure, p: &SolverParams) {
    update_beams(s, p);
    zero_and_beam_forces(s);
    collision::apply_terrain(&mut s.nodes);
    integrate(s, p);
}

/// Plastic deformation + breaking. For each beam, compare its current length to
/// its rest length: deviation beyond the elastic (yield) range permanently
/// shifts the rest length (a dent); deviation beyond the break threshold severs
/// the beam (it is skipped from then on and hidden by the renderer).
pub fn update_beams(s: &mut Structure, p: &SolverParams) {
    let n = &s.nodes;
    let b = &mut s.beams;
    for i in 0..b.len() {
        if b.broken[i] {
            continue;
        }
        let (ia, ib) = (b.a[i] as usize, b.b[i] as usize);
        let dx = n.px[ib] - n.px[ia];
        let dy = n.py[ib] - n.py[ia];
        let dz = n.pz[ib] - n.pz[ia];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let rest = b.rest[i];
        if rest < 1e-6 {
            continue;
        }
        let strain = (len - rest) / rest;

        if strain.abs() > b.break_strain[i] {
            b.broken[i] = true;
            continue;
        }
        // Plastic: keep `deform` worth of strain elastic; creep the rest length
        // toward the part of the deviation beyond yield.
        let yield_dev = rest * b.deform[i];
        let dev = len - rest;
        if dev.abs() > yield_dev {
            let excess = dev.abs() - yield_dev;
            b.rest[i] += excess * dev.signum() * p.plastic_rate;
        }
    }
}

/// Zero force accumulators and add all (unbroken) beam spring-damper forces.
/// Call this first each substep; vehicle/tire code then adds wheel forces before
/// `integrate`.
pub fn zero_and_beam_forces(s: &mut Structure) {
    let n = &mut s.nodes;
    for i in 0..n.len() {
        n.fx[i] = 0.0;
        n.fy[i] = 0.0;
        n.fz[i] = 0.0;
    }

    let b = &s.beams;
    for i in 0..b.len() {
        if b.broken[i] {
            continue;
        }
        let (ia, ib) = (b.a[i] as usize, b.b[i] as usize);
        let dx = n.px[ib] - n.px[ia];
        let dy = n.py[ib] - n.py[ia];
        let dz = n.pz[ib] - n.pz[ia];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-6 {
            continue;
        }
        let inv_len = 1.0 / len;
        let (nx, ny, nz) = (dx * inv_len, dy * inv_len, dz * inv_len);

        // Spring (Hooke) + damping along the beam axis.
        let spring = b.k[i] * (len - b.rest[i]);
        let rvx = n.vx[ib] - n.vx[ia];
        let rvy = n.vy[ib] - n.vy[ia];
        let rvz = n.vz[ib] - n.vz[ia];
        let rel_vel = rvx * nx + rvy * ny + rvz * nz;
        let f = spring + b.d[i] * rel_vel; // +f pulls a toward b

        let (fxv, fyv, fzv) = (f * nx, f * ny, f * nz);
        n.fx[ia] += fxv;
        n.fy[ia] += fyv;
        n.fz[ia] += fzv;
        n.fx[ib] -= fxv;
        n.fy[ib] -= fyv;
        n.fz[ib] -= fzv;
    }
}

/// Integrate node velocities/positions (semi-implicit Euler). Gravity is applied
/// as acceleration so pinned nodes (inv_mass == 0) are unaffected. Call after all
/// forces (beams + collision + wheel/tire) have been accumulated.
pub fn integrate(s: &mut Structure, p: &SolverParams) {
    let dt = p.substep_dt;
    let n = &mut s.nodes;
    let damp = 1.0 - p.global_damping * dt; // per-second drag, scaled by dt
    for i in 0..n.len() {
        let im = n.inv_mass[i];
        if im == 0.0 {
            n.vx[i] = 0.0;
            n.vy[i] = 0.0;
            n.vz[i] = 0.0;
            continue;
        }
        n.vx[i] = (n.vx[i] + n.fx[i] * im * dt) * damp;
        n.vy[i] = (n.vy[i] + (n.fy[i] * im + p.gravity) * dt) * damp;
        n.vz[i] = (n.vz[i] + n.fz[i] * im * dt) * damp;
        n.px[i] += n.vx[i] * dt;
        n.py[i] += n.vy[i] * dt;
        n.pz[i] += n.vz[i] * dt;
    }
}

/// Total mechanical energy (kinetic + gravitational + beam strain), used by tests
/// to confirm the solver is stable (bounded / non-exploding).
pub fn total_energy(s: &Structure, gravity: f32) -> f32 {
    let n = &s.nodes;
    let mut e = 0.0f32;
    for i in 0..n.len() {
        let im = n.inv_mass[i];
        if im == 0.0 {
            continue;
        }
        let m = 1.0 / im;
        let v2 = n.vx[i] * n.vx[i] + n.vy[i] * n.vy[i] + n.vz[i] * n.vz[i];
        e += 0.5 * m * v2; // kinetic
        e += m * (-gravity) * n.py[i]; // gravitational potential
    }
    let b = &s.beams;
    for i in 0..b.len() {
        if b.broken[i] {
            continue;
        }
        let (ia, ib) = (b.a[i] as usize, b.b[i] as usize);
        let dx = n.px[ib] - n.px[ia];
        let dy = n.py[ib] - n.py[ia];
        let dz = n.pz[ib] - n.pz[ia];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let stretch = len - b.rest[i];
        e += 0.5 * b.k[i] * stretch * stretch; // strain
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::softbody::structure::build_lattice;

    fn max_speed(s: &Structure) -> f32 {
        let n = &s.nodes;
        let mut m = 0.0f32;
        for i in 0..n.len() {
            m = m.max((n.vx[i] * n.vx[i] + n.vy[i] * n.vy[i] + n.vz[i] * n.vz[i]).sqrt());
        }
        m
    }

    /// A lattice dropped onto the flat spawn area must stay finite (stable solver)
    /// and come to rest above the ground (collision + damping work).
    #[test]
    fn lattice_drops_and_settles() {
        let spacing = 0.7;
        let mut s = build_lattice([4, 3, 4], spacing, [-1.0, 3.0, -1.0], 5.0, 60_000.0, 250.0);
        let p = SolverParams::default();

        // 4 seconds at 1 kHz.
        for _ in 0..4000 {
            substep(&mut s, &p);
        }

        // Everything finite (no explosion).
        for i in 0..s.nodes.len() {
            assert!(
                s.nodes.py[i].is_finite() && s.nodes.px[i].is_finite() && s.nodes.pz[i].is_finite(),
                "node {} went non-finite (solver unstable)",
                i
            );
        }
        // Settled (near zero motion).
        assert!(max_speed(&s) < 0.5, "structure did not settle: {}", max_speed(&s));
        // Resting on/above the ground (lowest node near radius above y=0).
        let min_y = s.nodes.py.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(min_y > -0.2 && min_y < 1.0, "unexpected resting height: {}", min_y);

        // Energy must be bounded (not blowing up).
        let e = total_energy(&s, p.gravity);
        assert!(e.is_finite(), "energy non-finite");
    }

    /// A single beam stretched past yield takes a permanent set; past the break
    /// threshold it severs.
    #[test]
    fn beam_yields_then_breaks() {
        use crate::softbody::structure::{BeamKind, Beams, Nodes};
        let mut nodes = Nodes::default();
        nodes.push([0.0, 0.0, 0.0], 1.0, 0.1);
        nodes.push([1.0, 0.0, 0.0], 1.0, 0.1);
        let mut beams = Beams::default();
        beams.push(0, 1, 1.0, 1000.0, 1.0, 0.1, 0.5, BeamKind::Normal); // yield 10%, break 50%
        let mut s = Structure::new(nodes, beams);
        let p = SolverParams::default();

        // Stretch to 1.3 (30% > 10% yield, < 50% break): should take a set.
        s.nodes.px[1] = 1.3;
        for _ in 0..50 {
            update_beams(&mut s, &p);
        }
        assert!(!s.beams.broken[0], "should not break at 30% strain");
        assert!(s.beams.rest[0] > 1.05, "rest length should creep up (set): {}", s.beams.rest[0]);

        // Now stretch past the break threshold.
        s.nodes.px[1] = 3.0;
        update_beams(&mut s, &p);
        assert!(s.beams.broken[0], "should break past the break strain");
    }

    /// Instrumented drop of the actual test body: confirms it MOVES while falling
    /// and WOBBLES after impact (deformation), rather than landing dead-still.
    /// Run with: `cargo test wobble -- --nocapture`.
    #[test]
    fn test_body_wobbles_after_impact() {
        use crate::softbody::SoftBody;
        let mut sb = SoftBody::test_lattice();
        let p = &sb.params;
        let g = p.gravity;

        // Beam-length spread = how deformed the body is vs. its rest shape.
        let deform_spread = |s: &Structure| {
            let mut max_strain = 0.0f32;
            for i in 0..s.beams.len() {
                let (ia, ib) = (s.beams.a[i] as usize, s.beams.b[i] as usize);
                let dx = s.nodes.px[ib] - s.nodes.px[ia];
                let dy = s.nodes.py[ib] - s.nodes.py[ia];
                let dz = s.nodes.pz[ib] - s.nodes.pz[ia];
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                max_strain = max_strain.max((len - s.beams.rest[i]).abs() / s.beams.rest[i]);
            }
            max_strain
        };

        let mut peak_fall_speed = 0.0f32;
        let mut peak_post_impact_deform = 0.0f32;
        // ~3 s at 60 fps, 1 kHz physics.
        for frame in 0..180 {
            sb.run(17);
            let ms = max_speed(&sb.structure);
            let df = deform_spread(&sb.structure);
            if frame < 40 {
                peak_fall_speed = peak_fall_speed.max(ms);
            } else {
                peak_post_impact_deform = peak_post_impact_deform.max(df);
            }
        }
        let e = total_energy(&sb.structure, g);
        println!(
            "peak_fall_speed={:.2} m/s  peak_post_impact_deform={:.3}  final_energy={:.1}",
            peak_fall_speed, peak_post_impact_deform, e
        );
        assert!(peak_fall_speed > 3.0, "body should fall (got {})", peak_fall_speed);
        assert!(
            peak_post_impact_deform > 0.02,
            "body should visibly deform on impact (got {})",
            peak_post_impact_deform
        );
        assert!(e.is_finite());
    }
}
