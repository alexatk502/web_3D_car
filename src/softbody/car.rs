//! The drivable soft-body car: a stiff box-frame chassis (node/beam) with four
//! wheel-hub nodes tied to the frame by softer suspension beams. Suspension,
//! body roll and weight transfer emerge from the beam network; the tire model
//! supplies grip and the drivetrain supplies torque.
//!
//! Local frame: forward = +X, up = +Y, right = +Z.

use crate::scene::{self, ObstacleBox};
use crate::softbody::drivetrain::Drivetrain;
use crate::softbody::solver::{self, SolverParams};
use crate::softbody::structure::{BeamKind, Beams, Nodes, Structure};
use crate::softbody::{collision, tire};
use glam::{Mat3, Quat, Vec3};
use std::collections::HashSet;

// --- Tunables ---
// Chassis node grid (real crumple zones). 5 x 3 x 3 = 45 nodes.
const CHASSIS_NX: usize = 5;
const CHASSIS_NY: usize = 3;
const CHASSIS_NZ: usize = 3;
const CAGE_MIN: [f32; 3] = [-1.4, 0.2, -0.7]; // body box (length, height, width)
const CAGE_MAX: [f32; 3] = [1.4, 0.8, 0.7];

// NOTE on damping: an interior grid node touches ~18 beams, so the *per-node*
// damping is ~18*CHASSIS_D. Explicit integration is stable only while
// 18*CHASSIS_D*inv_mass*dt < ~2, so heavier nodes are what let us damp harder.
const CHASSIS_K: f32 = 650_000.0; // structural stiffness (rigid to drive; ~5x original)
const CHASSIS_D: f32 = 600.0; // within the per-node stability budget at this mass
const CHASSIS_NODE_MASS: f32 = 12.0; // ~45*12 = 540 kg chassis (heavier = stiffer + dampable)
const CHASSIS_DEFORM: f32 = 0.14; // stays elastic under driving loads; dents past ~14% strain
const CHASSIS_BREAK: f32 = 0.55; // severs past 55% strain (hard crashes)

// Each wheel hub mounts to SUSP_MOUNTS spread-out chassis nodes (triangulated so
// the wheel can't pivot/wobble about a single cluster). The mounts are biased
// *inboard* (toward the car centre) so the suspension beams angle inward and
// locate the wheel laterally — stops it popping in/out of the body.
const SUSP_MOUNTS: usize = 3;
const SUSP_INBOARD_X: f32 = 0.55; // 0=at centre, 1=under the wheel (lengthwise)
const SUSP_INBOARD_Z: f32 = 0.20; // 0=at centreline, 1=under the wheel (sideways)
const SUSP_K: f32 = 48_000.0; // suspension spring (softer than chassis)
const SUSP_D: f32 = 1_900.0;
const SUSP_DEFORM: f32 = 0.35; // suspension flexes a lot before taking a set
const SUSP_BREAK: f32 = 1.20; // and is hard to snap off
const HUB_MASS: f32 = 20.0;

const WHEEL_RADIUS: f32 = 0.35;
const WHEEL_HALF_WIDTH: f32 = 0.18; // wheel render width (half)
const WHEEL_INERTIA: f32 = 1.2; // ~0.5*m*r^2

const MAX_STEER: f32 = 0.5; // radians
const STEER_RATE: f32 = 5.0;

const TIRE_MU: f32 = 1.7; // grippy arcade-sim
const WHEEL_CONTACT_K: f32 = 180_000.0;
const WHEEL_CONTACT_D: f32 = 4_000.0;

const BRAKE_TORQUE: f32 = 1_800.0; // per wheel
const HANDBRAKE_TORQUE: f32 = 3_200.0;
const REVERSE_TORQUE: f32 = 1_500.0;

const LIFT: f32 = WHEEL_RADIUS + 0.06; // spawn height so wheels rest on flat ground

pub struct Wheel {
    pub node: u32,
    pub radius: f32,
    pub steerable: bool,
    pub driven: bool,
    pub omega: f32, // spin (rad/s)
    pub spin: f32,  // accumulated spin angle (rad) for rendering
    pub inertia: f32,
    pub contact: bool,
}

pub struct Car {
    pub structure: Structure,
    pub params: SolverParams,
    pub wheels: Vec<Wheel>,
    pub drivetrain: Drivetrain,
    obstacles: Vec<ObstacleBox>,
    // Reference node groups used to derive the (deformable) car frame.
    front: Vec<u32>,
    rear: Vec<u32>,
    left: Vec<u32>,
    right: Vec<u32>,
    // Controls.
    steer: f32,
    throttle: f32,
    brake: f32,
    steer_in: f32,
    handbrake: bool,
}

impl Car {
    pub fn new() -> Self {
        let mut nodes = Nodes::default();
        let mut beams = Beams::default();

        // --- Chassis: a dense node grid (real crumple zones, BeamNG-style). ---
        // Grid index -> node id, so we can connect neighbours.
        let mut grid = vec![u32::MAX; CHASSIS_NX * CHASSIS_NY * CHASSIS_NZ];
        let gidx = |xi: usize, yi: usize, zi: usize| (xi * CHASSIS_NY + yi) * CHASSIS_NZ + zi;
        let lerp = |a: f32, b: f32, t: usize, n: usize| a + (b - a) * (t as f32 / (n - 1) as f32);

        let mut chassis = Vec::new();
        let (mut front, mut rear, mut left, mut right) = (vec![], vec![], vec![], vec![]);
        for xi in 0..CHASSIS_NX {
            for yi in 0..CHASSIS_NY {
                for zi in 0..CHASSIS_NZ {
                    let x = lerp(CAGE_MIN[0], CAGE_MAX[0], xi, CHASSIS_NX);
                    let y = lerp(CAGE_MIN[1], CAGE_MAX[1], yi, CHASSIS_NY);
                    let z = lerp(CAGE_MIN[2], CAGE_MAX[2], zi, CHASSIS_NZ);
                    let id = nodes.push([x, y, z], CHASSIS_NODE_MASS, 0.12);
                    grid[gidx(xi, yi, zi)] = id;
                    chassis.push(id);
                    // Frame groups: the extreme faces give robust forward/right axes.
                    if xi == CHASSIS_NX - 1 {
                        front.push(id);
                    } else if xi == 0 {
                        rear.push(id);
                    }
                    if zi == CHASSIS_NZ - 1 {
                        right.push(id);
                    } else if zi == 0 {
                        left.push(id);
                    }
                }
            }
        }
        // Connect each node to its forward 26-neighbourhood: axis beams (stiff),
        // face/body diagonals (softer) for shear/rigidity. Forward-only avoids dups.
        for xi in 0..CHASSIS_NX {
            for yi in 0..CHASSIS_NY {
                for zi in 0..CHASSIS_NZ {
                    let a = grid[gidx(xi, yi, zi)];
                    for dx in 0..=1usize {
                        for dy in 0..=1usize {
                            for dz in 0..=1usize {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let (nxi, nyi, nzi) = (xi + dx, yi + dy, zi + dz);
                                if nxi >= CHASSIS_NX || nyi >= CHASSIS_NY || nzi >= CHASSIS_NZ {
                                    continue;
                                }
                                let b = grid[gidx(nxi, nyi, nzi)];
                                let order = dx + dy + dz; // 1 axis, 2 face-diag, 3 body-diag
                                let k = if order == 1 { CHASSIS_K } else { CHASSIS_K * 0.6 };
                                connect(&mut nodes, &mut beams, a, b, k, CHASSIS_D, CHASSIS_DEFORM, CHASSIS_BREAK);
                            }
                        }
                    }
                }
            }
        }

        // --- Wheel hubs + suspension beams. ---
        let hub_local = [
            [1.1f32, 0.0, 0.75],   // FL (front, +z)
            [1.1, 0.0, -0.75],     // FR
            [-1.1, 0.0, 0.75],     // RL
            [-1.1, 0.0, -0.75],    // RR
        ];
        let mut wheels = Vec::new();
        for hl in hub_local {
            let hub = nodes.push(hl, HUB_MASS, WHEEL_RADIUS);
            nodes.mark_wheel(hub);
            // Mount to several spread-out chassis nodes (triangulated suspension).
            let mounts = suspension_mounts(&nodes, &chassis, hl, SUSP_MOUNTS);
            for c in mounts {
                connect(&mut nodes, &mut beams, hub, c, SUSP_K, SUSP_D, SUSP_DEFORM, SUSP_BREAK);
            }
            wheels.push(Wheel {
                node: hub,
                radius: WHEEL_RADIUS,
                steerable: hl[0] > 0.0, // front wheels steer
                driven: true,           // AWD for Phase 1 robustness
                omega: 0.0,
                spin: 0.0,
                inertia: WHEEL_INERTIA,
                contact: false,
            });
        }

        // Lift the whole car so the wheels sit on the (flat) spawn ground.
        let base = scene::terrain_height(0.0, 0.0) + LIFT;
        for i in 0..nodes.len() {
            nodes.py[i] += base;
        }

        let structure = Structure::new(nodes, beams);
        Car {
            structure,
            params: SolverParams::default(),
            wheels,
            drivetrain: Drivetrain::new(),
            obstacles: scene::obstacle_boxes(),
            front,
            rear,
            left,
            right,
            steer: 0.0,
            throttle: 0.0,
            brake: 0.0,
            steer_in: 0.0,
            handbrake: false,
        }
    }

    pub fn set_input(&mut self, throttle: f32, brake: f32, steer: f32, handbrake: bool, reset: bool) {
        if reset {
            self.reset();
            return;
        }
        self.throttle = throttle.clamp(0.0, 1.0);
        self.brake = brake.clamp(0.0, 1.0);
        self.steer_in = steer.clamp(-1.0, 1.0);
        self.handbrake = handbrake;
    }

    pub fn reset(&mut self) {
        self.structure.reset();
        for w in &mut self.wheels {
            w.omega = 0.0;
            w.contact = false;
        }
        self.drivetrain.reset();
        self.steer = 0.0;
    }

    pub fn run(&mut self, substeps: u32) {
        for _ in 0..substeps {
            self.substep();
        }
    }

    fn substep(&mut self) {
        let dt = self.params.substep_dt;

        // Smooth steering toward the target angle.
        let target = self.steer_in * MAX_STEER;
        self.steer += (target - self.steer) * (STEER_RATE * dt).min(1.0);

        // Frame + body velocity (read before mutable borrows).
        let (fwd, right, up) = self.frame();
        let v_long_body = self.avg_velocity().dot(fwd);

        // Plastic deformation + breaking (uses last substep's positions).
        solver::update_beams(&mut self.structure, &self.params);

        // Build the set of currently-connected node pairs (for self-collision),
        // before the mutable node borrow.
        let mut connected: HashSet<(u32, u32)> = HashSet::new();
        for i in 0..self.structure.beams.len() {
            if !self.structure.beams.broken[i] {
                let (a, b) = (self.structure.beams.a[i], self.structure.beams.b[i]);
                connected.insert((a.min(b), a.max(b)));
            }
        }

        // Beam forces + collisions for chassis nodes (wheels skip terrain/self).
        solver::zero_and_beam_forces(&mut self.structure);
        collision::apply_terrain(&mut self.structure.nodes);
        collision::apply_obstacles(&mut self.structure.nodes, &self.obstacles);
        collision::apply_self_collision(&mut self.structure.nodes, |i, j| {
            connected.contains(&(i as u32, j as u32))
        });

        // Drivetrain: torque from average driven-wheel speed.
        let (mut sum, mut nd) = (0.0f32, 0u32);
        for w in &self.wheels {
            if w.driven {
                sum += w.omega;
                nd += 1;
            }
        }
        let avg_omega = if nd > 0 { sum / nd as f32 } else { 0.0 };
        let total_drive = self.drivetrain.update(self.throttle, avg_omega, dt);
        let per_drive = if nd > 0 { total_drive / nd as f32 } else { 0.0 };

        let reversing = self.throttle < 0.05 && self.brake > 0.05 && v_long_body < 1.0;

        // Snapshot scalar controls for the loop (avoid borrow conflicts).
        let (steer, brake, handbrake) = (self.steer, self.brake, self.handbrake);
        let nodes = &mut self.structure.nodes;
        let wheels = &mut self.wheels;

        for w in wheels.iter_mut() {
            let i = w.node as usize;
            let p = Vec3::new(nodes.px[i], nodes.py[i], nodes.pz[i]);
            let v = Vec3::new(nodes.vx[i], nodes.vy[i], nodes.vz[i]);

            // Wheel heading (steered) + ground-plane lateral axis.
            let s = if w.steerable { steer } else { 0.0 };
            let heading = (fwd * s.cos() + right * s.sin()).normalize_or_zero();
            let lat = up.cross(heading).normalize_or_zero();
            let v_long = v.dot(heading);
            let v_lat = v.dot(lat);

            // Per-wheel drive / brake.
            let mut drive_tq = if w.driven { per_drive } else { 0.0 };
            let mut brake_tq = 0.0;
            if reversing {
                if w.driven {
                    drive_tq = -REVERSE_TORQUE;
                }
            } else if brake > 0.05 {
                brake_tq = BRAKE_TORQUE * brake;
            }
            if handbrake && !w.steerable {
                brake_tq += HANDBRAKE_TORQUE;
            }

            // Ground contact + tire forces.
            let ground = scene::terrain_height(p.x, p.z);
            let pen = (ground + w.radius) - p.y;
            let mut fx = 0.0;
            if pen > 0.0 {
                let nn = scene::terrain_normal(p.x, p.z);
                let nrm = Vec3::new(nn[0], nn[1], nn[2]);
                let vn = v.dot(nrm);
                let fz = (WHEEL_CONTACT_K * pen - WHEEL_CONTACT_D * vn).max(0.0);

                let denom = v_long.abs() + 1.0; // +1 tames low-speed singularities
                let kappa = (w.omega * w.radius - v_long) / denom;
                let alpha = (v_lat / denom).atan();
                let lo = tire::longitudinal(fz, TIRE_MU, kappa);
                let la = tire::lateral(fz, TIRE_MU, alpha);
                let (fxx, fyy) = tire::friction_circle(lo, la, TIRE_MU * fz);
                fx = fxx;

                let force = nrm * fz + heading * fxx + lat * fyy;
                nodes.fx[i] += force.x;
                nodes.fy[i] += force.y;
                nodes.fz[i] += force.z;
                w.contact = true;
            } else {
                w.contact = false;
            }

            // Wheel spin: I·dω = drive − road_reaction; then brake toward zero.
            let road_reaction = fx * w.radius;
            w.omega += (drive_tq - road_reaction) / w.inertia * dt;
            if brake_tq > 0.0 {
                let dw = brake_tq / w.inertia * dt;
                if w.omega.abs() <= dw {
                    w.omega = 0.0;
                } else {
                    w.omega -= dw * w.omega.signum();
                }
            }

            // Accumulate the visual spin angle.
            w.spin += w.omega * dt;
        }

        solver::integrate(&mut self.structure, &self.params);
    }

    /// Current car frame (forward, right, up) from the deformable node groups.
    fn frame(&self) -> (Vec3, Vec3, Vec3) {
        let n = &self.structure.nodes;
        let fc = centroid(n, &self.front);
        let rc = centroid(n, &self.rear);
        let rightc = centroid(n, &self.right);
        let leftc = centroid(n, &self.left);
        let fwd = (fc - rc).normalize_or_zero();
        let right_raw = rightc - leftc;
        let up = right_raw.cross(fwd).normalize_or_zero();
        let right = fwd.cross(up).normalize_or_zero();
        let fwd = if fwd.length_squared() > 0.0 { fwd } else { Vec3::X };
        (fwd, right, up)
    }

    fn avg_velocity(&self) -> Vec3 {
        let n = &self.structure.nodes;
        let count = n.len().max(1) as f32;
        let mut v = Vec3::ZERO;
        for i in 0..n.len() {
            v += Vec3::new(n.vx[i], n.vy[i], n.vz[i]);
        }
        v / count
    }

    pub fn centroid(&self) -> Vec3 {
        let n = &self.structure.nodes;
        let count = n.len().max(1) as f32;
        let mut c = Vec3::ZERO;
        for i in 0..n.len() {
            c += Vec3::new(n.px[i], n.py[i], n.pz[i]);
        }
        c / count
    }

    pub fn forward(&self) -> Vec3 {
        self.frame().0
    }

    pub fn speed_kmh(&self) -> f32 {
        self.avg_velocity().dot(self.frame().0).abs() * 3.6
    }

    pub fn rpm(&self) -> f32 {
        self.drivetrain.rpm
    }

    pub fn gear(&self) -> u32 {
        self.drivetrain.gear as u32 + 1
    }

    pub fn node_count(&self) -> usize {
        self.structure.nodes.len()
    }

    pub fn wheel_count(&self) -> usize {
        self.wheels.len()
    }

    /// Wheel render dimensions (radius, half-width). The cylinder mesh is built
    /// to these on the JS side.
    pub fn wheel_dims(&self) -> (f32, f32) {
        (WHEEL_RADIUS, WHEEL_HALF_WIDTH)
    }

    /// Body box extents (used to author the body hull shape).
    pub fn cage(&self) -> ([f32; 3], [f32; 3]) {
        (CAGE_MIN, CAGE_MAX)
    }

    /// Number of chassis (non-wheel) nodes — the first `chassis_count` nodes.
    pub fn chassis_count(&self) -> usize {
        self.structure.nodes.len() - self.wheels.len()
    }

    /// Rest (spawn) positions of the chassis nodes, flat [x,y,z, ...]. Used by JS
    /// to build the body-mesh skinning weights.
    pub fn chassis_rest(&self) -> Vec<f32> {
        let n = self.chassis_count();
        let s = &self.structure;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            out.push(s.spawn_px[i]);
            out.push(s.spawn_py[i]);
            out.push(s.spawn_pz[i]);
        }
        out
    }

    /// World transform (position, rotation) of wheel `i` for rendering. Follows
    /// the hub node (so suspension travel shows), steers (front), and spins.
    /// The cylinder mesh axis is local +Z (the axle).
    pub fn wheel_transform(&self, i: usize) -> (Vec3, Quat) {
        let (fwd, right, _up) = self.frame();
        let w = &self.wheels[i];
        let s = if w.steerable { self.steer } else { 0.0 };
        let (sin, cos) = s.sin_cos();
        let heading = (fwd * cos + right * sin).normalize_or_zero();
        let axle = (right * cos - fwd * sin).normalize_or_zero();
        let up_w = axle.cross(heading).normalize_or_zero();
        // Basis: local X = heading, local Y = up_w, local Z = axle (right-handed).
        let base = Quat::from_mat3(&Mat3::from_cols(heading, up_w, axle));
        let rot = base * Quat::from_rotation_z(w.spin);
        let ni = w.node as usize;
        let n = &self.structure.nodes;
        (Vec3::new(n.px[ni], n.py[ni], n.pz[ni]), rot)
    }
}

fn centroid(n: &Nodes, group: &[u32]) -> Vec3 {
    if group.is_empty() {
        return Vec3::ZERO;
    }
    let mut c = Vec3::ZERO;
    for &g in group {
        let i = g as usize;
        c += Vec3::new(n.px[i], n.py[i], n.pz[i]);
    }
    c / group.len() as f32
}

#[allow(clippy::too_many_arguments)]
fn connect(
    nodes: &mut Nodes,
    beams: &mut Beams,
    a: u32,
    b: u32,
    k: f32,
    d: f32,
    deform: f32,
    break_strain: f32,
) {
    let (ia, ib) = (a as usize, b as usize);
    let dx = nodes.px[ia] - nodes.px[ib];
    let dy = nodes.py[ia] - nodes.py[ib];
    let dz = nodes.pz[ia] - nodes.pz[ib];
    let rest = (dx * dx + dy * dy + dz * dz).sqrt();
    beams.push(a, b, rest, k, d, deform, break_strain, BeamKind::Normal);
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1 frame at 60 fps ~= 17 substeps at 1 kHz.
    fn frames(car: &mut Car, n: u32) {
        for _ in 0..n {
            car.run(17);
        }
    }

    #[test]
    fn car_settles_upright_on_its_wheels() {
        let mut car = Car::new();
        car.set_input(0.0, 0.0, 0.0, false, false);
        frames(&mut car, 120); // ~2 s

        let (_f, _r, up) = car.frame();
        assert!(up.y > 0.7, "car should stay upright (up.y={})", up.y);
        assert!(car.speed_kmh() < 3.0, "car should be at rest (got {} km/h)", car.speed_kmh());

        // All wheels in contact, resting near the ground.
        let in_contact = car.wheels.iter().filter(|w| w.contact).count();
        assert!(in_contact >= 3, "wheels should be on the ground ({} in contact)", in_contact);
        for i in 0..car.structure.nodes.len() {
            assert!(car.structure.nodes.py[i].is_finite());
        }
    }

    #[test]
    fn car_accelerates_forward_then_brakes() {
        let mut car = Car::new();
        car.set_input(0.0, 0.0, 0.0, false, false);
        frames(&mut car, 60); // settle

        let start = car.centroid();
        car.set_input(1.0, 0.0, 0.0, false, false); // full throttle
        frames(&mut car, 180); // ~3 s
        let v = car.speed_kmh();
        assert!(v > 10.0, "car should accelerate under throttle (got {} km/h)", v);

        // Moved roughly along +X (forward).
        let moved = car.centroid() - start;
        assert!(moved.x > 3.0, "car should move forward (+X), moved.x={}", moved.x);
        assert!(car.rpm() > 900.0 && car.rpm().is_finite());

        // Braking slows it down. Capture the minimum speed over the window, since
        // holding the brake past a stop intentionally engages auto-reverse.
        car.set_input(0.0, 1.0, 0.0, false, false);
        let mut min_speed = v;
        for _ in 0..60 {
            car.run(17);
            min_speed = min_speed.min(car.speed_kmh());
        }
        assert!(min_speed < v * 0.5, "brakes should clearly slow the car (min {} vs {})", min_speed, v);
    }

    #[test]
    fn crashing_into_obstacle_deforms_the_car() {
        let mut car = Car::new();
        frames(&mut car, 30); // settle

        // Fire the whole car at the nearest obstacle box.
        let box0 = &car.obstacles[0];
        let target = Vec3::new(box0.center[0], box0.center[1], box0.center[2]);
        let dir = (target - car.centroid()).normalize();
        let v = dir * 32.0;
        for i in 0..car.structure.nodes.len() {
            car.structure.nodes.vx[i] = v.x;
            car.structure.nodes.vy[i] = v.y;
            car.structure.nodes.vz[i] = v.z;
        }

        frames(&mut car, 120); // ~2 s of flight + impact

        // At least one beam should have taken a permanent set or broken.
        let mut deformed = 0;
        let mut broken = 0;
        for i in 0..car.structure.beams.len() {
            if car.structure.beams.broken[i] {
                broken += 1;
            } else {
                let drift = (car.structure.beams.rest[i] - car.structure.spawn_rest[i]).abs();
                if drift / car.structure.spawn_rest[i] > 0.02 {
                    deformed += 1;
                }
            }
        }
        assert!(
            deformed + broken > 0,
            "crash should deform or break beams (deformed={}, broken={})",
            deformed,
            broken
        );

        // Reset must restore the pristine shape.
        car.reset();
        for i in 0..car.structure.beams.len() {
            assert!(!car.structure.beams.broken[i], "reset should un-break beams");
            assert!((car.structure.beams.rest[i] - car.structure.spawn_rest[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn car_steers() {
        let mut car = Car::new();
        frames(&mut car, 60);
        car.set_input(0.8, 0.0, 0.0, false, false);
        frames(&mut car, 120); // get moving
        let h0 = car.forward();
        car.set_input(0.8, 0.0, 1.0, false, false); // steer
        frames(&mut car, 120);
        let h1 = car.forward();
        let yaw_change = (h1.z.atan2(h1.x) - h0.z.atan2(h0.x)).abs();
        assert!(yaw_change > 0.05, "steering should change heading (Δyaw={})", yaw_change);
    }
}

/// Pick `count` chassis nodes near a hub but spread apart and biased inboard.
/// The mounts form a triangulated suspension that locates the wheel laterally
/// (inboard bias prevents wheels popping in/out of the body).
fn suspension_mounts(nodes: &Nodes, chassis: &[u32], hub_local: [f32; 3], count: usize) -> Vec<u32> {
    let d2 = |i: usize, p: [f32; 3]| {
        let dx = nodes.px[i] - p[0];
        let dy = nodes.py[i] - p[1];
        let dz = nodes.pz[i] - p[2];
        dx * dx + dy * dy + dz * dz
    };

    // Bias the search target inboard (toward car centre at [0, hub_local[1], 0]).
    let biased_target = [
        hub_local[0] * (1.0 - SUSP_INBOARD_X),
        hub_local[1],
        hub_local[2] * (1.0 - SUSP_INBOARD_Z),
    ];

    let mut scored: Vec<(f32, u32)> = chassis
        .iter()
        .map(|&c| (d2(c as usize, biased_target), c))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let pool: Vec<u32> = scored
        .iter()
        .take((count * 3).min(scored.len()))
        .map(|&(_, c)| c)
        .collect();

    let mut picked = vec![pool[0]]; // nearest inboard node first
    while picked.len() < count && picked.len() < pool.len() {
        // Add the pool node that is farthest from everything already picked.
        let mut best = pool[0];
        let mut best_sep = -1.0f32;
        for &c in &pool {
            if picked.contains(&c) {
                continue;
            }
            let ci = c as usize;
            let mut min_sep = f32::INFINITY;
            for &p in &picked {
                let pj = p as usize;
                min_sep = min_sep.min(d2(ci, [nodes.px[pj], nodes.py[pj], nodes.pz[pj]]));
            }
            if min_sep > best_sep {
                best_sep = min_sep;
                best = c;
            }
        }
        picked.push(best);
    }
    picked
}
