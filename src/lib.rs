//! WASM physics core. Phase 0 of the soft-body (BeamNG-style) migration.
//!
//! `World` owns the static scene (terrain + obstacles, rendered via per-object
//! model matrices) and a `SoftBody` (node/beam mass-spring). JS drives it: push
//! input, step with a real dt, then read two zero-copy buffers from linear
//! memory — the static render buffer (view + model matrices) and the soft-body
//! node-position buffer (for the debug node/beam renderer).

mod camera;
mod physics;
mod render_state;
mod scene;
mod softbody;

use camera::{CamMode, Camera};
use glam::{Mat4, Quat, Vec3};
use physics::Physics;
use rapier3d::prelude::*;
use render_state::RenderBuffer;
use scene::descriptor_json;
use softbody::car::Car;
use wasm_bindgen::prelude::*;

// Browser thread-pool initializer (Web Workers + SharedArrayBuffer). JS must
// `await initThreadPool(navigator.hardwareConcurrency)` once before stepping so
// the parallel solver has workers. wasm-only; native uses rayon's OS pool.
#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Driver/dev input. (Driving is wired up in Phase 1; Phase 0 only uses `reset`.)
#[derive(Clone, Copy, Default)]
struct Input {
    throttle: f32,
    brake: f32,
    steer: f32,
    handbrake: bool,
    reset: bool,
    clutch: f32, // 0 = engaged, 1 = fully disengaged (manual mode)
}

/// Cap on solver substeps per render frame (anti-spiral after a tab stall).
const MAX_SUBSTEPS_PER_FRAME: u32 = 64;

#[wasm_bindgen]
pub struct World {
    physics: Physics,
    cars: Vec<Car>,
    active: usize, // the car the camera follows / input drives
    camera: Camera,
    static_handles: Vec<RigidBodyHandle>,
    descriptor: String,
    soft_descriptor: String,
    car_descriptor: String,
    num_static: usize,  // terrain + obstacles + surface patches
    num_objects: usize, // num_static + 4 wheels per car (total model matrices)
    node_offsets: Vec<usize>, // per-car node start index (in nodes), prefix sum
    buffer: RenderBuffer,
    node_buf: Vec<f32>, // [x,y,z, ...] per node across all cars, refreshed each frame
    line_buf: Vec<u32>, // active beam endpoint pairs (global node indices), by stress band
    line_count: usize,  // number of valid indices in line_buf this frame
    line_bands: [usize; 4], // index count per stress band (green→yellow→orange→red)
    input: Input,
    accumulator: f32,
    last_substeps: u32, // substeps run on the most recent frame (HUD)
    // Impact detection for crash audio (tracks the active car).
    prev_speed_ms: f32,
    prev_broken: usize,
    last_impact: f32,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new() -> World {
        console_error_panic_hook::set_once();

        // Static scene (terrain + obstacles). Rapier holds them but is not stepped
        // — the soft body collides with the terrain analytically.
        let mut physics = Physics::new();
        let (static_handles, descs) = scene::build_static(&mut physics);
        let num_static = descs.len();
        let descriptor = descriptor_json(&descs);

        let car = Car::new();
        let car_descriptor = car_descriptor_json(&car);

        let mut world = World {
            physics,
            cars: vec![car],
            active: 0,
            camera: Camera::new(),
            static_handles,
            descriptor,
            soft_descriptor: String::new(),
            car_descriptor,
            num_static,
            num_objects: 0,
            node_offsets: Vec::new(),
            buffer: RenderBuffer::new(num_static),
            node_buf: Vec::new(),
            line_buf: Vec::new(),
            line_count: 0,
            line_bands: [0; 4],
            input: Input::default(),
            accumulator: 0.0,
            last_substeps: 0,
            prev_speed_ms: 0.0,
            prev_broken: 0,
            last_impact: 0.0,
        };
        world.rebuild_layout();
        world
    }

    /// Recompute all per-car-count layout: object count, node offsets, buffer
    /// sizes, the render buffer, and the soft descriptor. Called at startup and
    /// after spawning a car. JS must re-read sizes/descriptors afterwards.
    fn rebuild_layout(&mut self) {
        let mut offsets = Vec::with_capacity(self.cars.len());
        let mut total_nodes = 0usize;
        let mut total_beams = 0usize;
        for c in &self.cars {
            offsets.push(total_nodes);
            total_nodes += c.node_count();
            total_beams += c.structure.beams.len();
        }
        self.node_offsets = offsets;
        self.num_objects = self.num_static + self.cars.iter().map(|c| c.wheel_count()).sum::<usize>();
        self.node_buf = vec![0.0; total_nodes * 3];
        self.line_buf = vec![0u32; total_beams * 2];
        self.buffer = RenderBuffer::new(self.num_objects);
        self.soft_descriptor = soft_descriptor_json(&self.cars, &self.node_offsets);
    }

    /// JSON describing the static renderable objects (kind, dims, color), in the
    /// same order as the model matrices in the shared buffer.
    #[wasm_bindgen(getter)]
    pub fn descriptor(&self) -> String {
        self.descriptor.clone()
    }

    /// JSON describing the soft body for the debug renderer:
    /// `{ "nodeCount": N, "beams": [a,b, ...], "color": [r,g,b] }`.
    #[wasm_bindgen(getter)]
    pub fn soft_descriptor(&self) -> String {
        self.soft_descriptor.clone()
    }

    /// JSON describing the car render assets: wheel dims/color, body color, and
    /// the chassis FFD cage extents used to skin the body mesh.
    #[wasm_bindgen(getter)]
    pub fn car_descriptor(&self) -> String {
        self.car_descriptor.clone()
    }

    /// Total wheel model matrices across all cars (4 per car).
    pub fn wheel_count(&self) -> usize {
        self.cars.iter().map(|c| c.wheel_count()).sum()
    }

    /// Number of cars currently in the world.
    pub fn car_count(&self) -> usize {
        self.cars.len()
    }

    /// Node-buffer start index (in nodes) of car `c`, so JS can skin each body to
    /// its own chassis slice.
    pub fn car_node_offset(&self, c: usize) -> usize {
        self.node_offsets.get(c).copied().unwrap_or(0)
    }

    /// Spawn another car at world (x, z) with heading `yaw`. Grows the buffers;
    /// JS must re-read sizes/descriptors and rebuild its meshes afterwards.
    pub fn spawn_car(&mut self, x: f32, z: f32, yaw: f32) {
        let mut car = Car::new_at(Vec3::new(x, 0.0, z), yaw);
        car.set_threaded(self.cars.first().map(|c| c.is_threaded()).unwrap_or(false));
        self.cars.push(car);
        self.rebuild_layout();
    }

    /// Spawn a car relative to the active car's frame: `ahead` metres forward and
    /// `side` metres to its right, facing the same heading. Capped at 8 cars.
    pub fn spawn_near_active(&mut self, ahead: f32, side: f32) {
        if self.cars.len() >= 8 {
            return;
        }
        let c = self.cars[self.active].centroid();
        let f = self.cars[self.active].forward();
        let right = Vec3::new(-f.z, 0.0, f.x); // forward rotated -90° about +Y
        let pos = c + f * ahead + right * side;
        let yaw = f.z.atan2(f.x);
        self.spawn_car(pos.x, pos.z, yaw);
    }

    /// Which car the camera follows and input drives.
    pub fn set_active(&mut self, idx: usize) {
        if idx < self.cars.len() {
            self.active = idx;
            // Reset impact tracking to the new car so switching doesn't spike audio.
            self.prev_speed_ms = self.cars[idx].avg_speed_ms();
            self.prev_broken = self.cars[idx].broken_beam_count();
        }
    }
    pub fn active_car(&self) -> usize {
        self.active
    }

    /// Enable the parallel (multi-threaded) solver path on all cars. Call from JS
    /// only after `initThreadPool` has resolved (cross-origin isolated + pool up).
    pub fn set_threaded(&mut self, on: bool) {
        for c in &mut self.cars {
            c.set_threaded(on);
        }
    }

    /// Set the current input. `steer` is -1..1 (positive = left). `clutch` is
    /// 0 (engaged) .. 1 (disengaged) and only matters in manual mode.
    #[allow(clippy::too_many_arguments)]
    pub fn set_input(
        &mut self,
        throttle: f32,
        brake: f32,
        steer: f32,
        handbrake: bool,
        reset: bool,
        clutch: f32,
    ) {
        self.input = Input {
            throttle,
            brake,
            steer,
            handbrake,
            reset,
            clutch,
        };
    }

    /// Manual gearbox of the active car: sequential shift up/down + mode toggle.
    pub fn shift_up(&mut self) {
        self.cars[self.active].shift_up();
    }
    pub fn shift_down(&mut self) {
        self.cars[self.active].shift_down();
    }
    pub fn toggle_manual(&mut self) {
        self.cars[self.active].toggle_manual();
    }
    pub fn is_manual(&self) -> bool {
        self.cars[self.active].is_manual()
    }
    /// Clutch engagement 0..1 (1 = locked) of the active car, for the HUD.
    pub fn clutch(&self) -> f32 {
        self.cars[self.active].clutch_engagement()
    }

    /// Advance by `dt` real seconds (substep accumulator), then recompute the
    /// camera and refill the render + node buffers.
    pub fn step(&mut self, dt: f32, camera_mode: u32, orbit_dx: f32, orbit_dy: f32) {
        // Drive the active car; the others coast (zero input). Reset (R) only
        // affects the active car (handled inside set_input).
        let active = self.active;
        for (i, c) in self.cars.iter_mut().enumerate() {
            if i == active {
                c.set_input(
                    self.input.throttle,
                    self.input.brake,
                    self.input.steer,
                    self.input.handbrake,
                    self.input.reset,
                    self.input.clutch,
                );
            } else {
                c.set_input(0.0, 0.0, 0.0, false, false, 0.0);
            }
        }

        let substep_dt = self.cars[0].params.substep_dt;
        self.accumulator += dt.min(0.1);
        let mut steps = 0u32;
        while self.accumulator >= substep_dt && steps < MAX_SUBSTEPS_PER_FRAME {
            self.world_substep();
            self.accumulator -= substep_dt;
            steps += 1;
        }
        // Drop any leftover backlog so we don't spiral after a stall.
        if self.accumulator > substep_dt {
            self.accumulator = 0.0;
        }
        self.last_substeps = steps;

        // Impact level for crash audio (active car): sudden speed drop + beams snapped.
        let now_speed = self.cars[active].avg_speed_ms();
        let speed_drop = (self.prev_speed_ms - now_speed).max(0.0);
        self.prev_speed_ms = now_speed;
        let broken_now = self.cars[active].broken_beam_count();
        let broke_delta = broken_now.saturating_sub(self.prev_broken);
        self.prev_broken = broken_now;
        self.last_impact = broke_delta as f32 * 2.0 + speed_drop;

        self.refill_buffer(camera_mode, dt, orbit_dx, orbit_dy);
        self.refill_nodes();
        self.refill_lines();
    }

    /// One physics substep across all cars: accumulate per-car forces, inject
    /// cross-vehicle collision forces, then integrate all cars together.
    fn world_substep(&mut self) {
        for c in &mut self.cars {
            c.accumulate_forces();
        }
        self.cross_car_collision();
        for c in &mut self.cars {
            c.integrate();
        }
    }

    /// Vehicle-vehicle collision with an AABB broadphase. Inflated by ~the max node
    /// radius so contacts aren't missed at the boundary. Skipped for a single car.
    fn cross_car_collision(&mut self) {
        let n = self.cars.len();
        if n < 2 {
            return;
        }
        const MARGIN: f32 = 0.5;
        let aabbs: Vec<(Vec3, Vec3)> = self.cars.iter().map(|c| c.aabb()).collect();
        for a in 0..n {
            for b in (a + 1)..n {
                let (alo, ahi) = aabbs[a];
                let (blo, bhi) = aabbs[b];
                let apart = ahi.x + MARGIN < blo.x
                    || bhi.x + MARGIN < alo.x
                    || ahi.y + MARGIN < blo.y
                    || bhi.y + MARGIN < alo.y
                    || ahi.z + MARGIN < blo.z
                    || bhi.z + MARGIN < alo.z;
                if apart {
                    continue;
                }
                // Disjoint &mut to two cars via split_at_mut (a < b).
                let (left, right) = self.cars.split_at_mut(b);
                softbody::collision::cross_body_collision(
                    &mut left[a].structure.nodes,
                    &mut right[0].structure.nodes,
                );
            }
        }
    }

    fn refill_buffer(&mut self, camera_mode: u32, dt: f32, orbit_dx: f32, orbit_dy: f32) {
        // Camera tracks the ACTIVE car's centroid, oriented to its heading.
        let target = self.cars[self.active].centroid();
        let fwd = self.cars[self.active].forward();
        // Build a yaw-only rotation from the car's heading so chase/hood cameras
        // sit behind the car (the camera expects forward = local +X).
        let yaw = fwd.z.atan2(fwd.x);
        let rot = Quat::from_rotation_y(-yaw);
        let view = self.camera.view(
            CamMode::from_u32(camera_mode),
            target,
            rot,
            dt.max(1e-4),
            orbit_dx,
            orbit_dy,
        );
        self.buffer.set_view(&view);

        // Static objects (terrain + obstacles + patches) by their fixed transforms.
        for (idx, h) in self.static_handles.iter().enumerate() {
            let iso = self.physics.bodies[*h].position();
            self.buffer.set_model(idx, &iso_to_mat4(iso));
        }
        // Wheels: each car's 4 hub-following matrices, laid out contiguously after
        // the static objects: [car0 w0..w3][car1 w0..w3]...
        let mut slot = self.num_static;
        for car in &self.cars {
            for i in 0..car.wheel_count() {
                let (pos, rot) = car.wheel_transform(i);
                self.buffer
                    .set_model(slot, &Mat4::from_rotation_translation(rot, pos));
                slot += 1;
            }
        }
    }

    fn refill_nodes(&mut self) {
        for (c, car) in self.cars.iter().enumerate() {
            let off = self.node_offsets[c];
            let n = &car.structure.nodes;
            for i in 0..n.len() {
                let k = (off + i) * 3;
                self.node_buf[k] = n.px[i];
                self.node_buf[k + 1] = n.py[i];
                self.node_buf[k + 2] = n.pz[i];
            }
        }
    }

    /// Fill `line_buf` with the endpoint pairs of unbroken beams, ordered by stress
    /// band (broken beams vanish). Stress = current strain relative to the beam's
    /// own break threshold (0..1), bucketed into 4 bands so the renderer can draw a
    /// green→yellow→orange→red heatmap with one colored draw call per band.
    fn refill_lines(&mut self) {
        // Per-beam stress band using a car's own nodes.
        let band_of = |car: &Car, i: usize| -> usize {
            let b = &car.structure.beams;
            let n = &car.structure.nodes;
            let (ia, ib) = (b.a[i] as usize, b.b[i] as usize);
            let dx = n.px[ib] - n.px[ia];
            let dy = n.py[ib] - n.py[ia];
            let dz = n.pz[ib] - n.pz[ia];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            let rest = b.rest[i];
            if rest < 1e-6 {
                return 0;
            }
            let strain = (len - rest).abs() / rest;
            let frac = (strain / b.break_strain[i].max(1e-3)).clamp(0.0, 1.0);
            ((frac * 4.0) as usize).min(3)
        };

        // Pass 1: count active beams per band across ALL cars.
        let mut counts = [0usize; 4];
        for car in &self.cars {
            let b = &car.structure.beams;
            for i in 0..b.len() {
                if !b.broken[i] {
                    counts[band_of(car, i)] += 1;
                }
            }
        }
        // Band start cursors (in pairs).
        let mut cursor = [0usize; 4];
        let mut acc = 0;
        for band in 0..4 {
            cursor[band] = acc;
            acc += counts[band];
        }
        // Pass 2: place each beam's endpoint pair (GLOBAL node indices) into its band.
        for (c, car) in self.cars.iter().enumerate() {
            let off = self.node_offsets[c] as u32;
            let b = &car.structure.beams;
            for i in 0..b.len() {
                if b.broken[i] {
                    continue;
                }
                let band = band_of(car, i);
                let k = cursor[band] * 2;
                self.line_buf[k] = b.a[i] + off;
                self.line_buf[k + 1] = b.b[i] + off;
                cursor[band] += 1;
            }
        }
        self.line_count = acc * 2;
        self.line_bands = [counts[0] * 2, counts[1] * 2, counts[2] * 2, counts[3] * 2];
    }

    // --- Static render buffer (view + model matrices) ---
    pub fn buffer_ptr(&self) -> *const f32 {
        self.buffer.ptr()
    }
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
    pub fn object_count(&self) -> usize {
        self.num_objects
    }

    // --- Soft-body node positions (for the debug renderer) ---
    pub fn node_buffer_ptr(&self) -> *const f32 {
        self.node_buf.as_ptr()
    }
    pub fn node_buffer_len(&self) -> usize {
        self.node_buf.len()
    }
    /// Total nodes across all cars.
    pub fn node_count(&self) -> usize {
        self.cars.iter().map(|c| c.node_count()).sum()
    }

    /// Total beam count across all cars (including broken), for the HUD.
    pub fn beam_count(&self) -> usize {
        self.cars.iter().map(|c| c.structure.beams.len()).sum()
    }

    /// Substeps executed on the most recent frame (HUD).
    pub fn substeps_last_frame(&self) -> u32 {
        self.last_substeps
    }

    // --- Active (unbroken) beam line indices (for the debug renderer) ---
    pub fn line_buffer_ptr(&self) -> *const u32 {
        self.line_buf.as_ptr()
    }
    pub fn line_buffer_len(&self) -> usize {
        self.line_buf.len()
    }
    /// Number of valid indices in the line buffer this frame (2 per unbroken beam).
    pub fn line_count(&self) -> usize {
        self.line_count
    }

    /// Index count per stress band [green, yellow, orange, red]. `line_buf` is
    /// ordered by band, so the renderer draws each band as a contiguous colored
    /// segment (a strain heatmap). Sums to `line_count`.
    pub fn line_band_counts(&self) -> Vec<u32> {
        self.line_bands.iter().map(|&c| c as u32).collect()
    }

    /// Active car forward speed in km/h.
    pub fn speed_kmh(&self) -> f32 {
        self.cars[self.active].speed_kmh()
    }

    /// Active car engine RPM (HUD).
    pub fn rpm(&self) -> f32 {
        self.cars[self.active].rpm()
    }

    /// Impact intensity this frame (0 ≈ calm; spikes on crashes/landings). Drives
    /// the crash/thud audio. Unitless: ~speed-drop in m/s plus 2× beams snapped.
    pub fn impact_level(&self) -> f32 {
        self.last_impact
    }

    /// Active car gear: -1 = reverse, 0 = neutral, 1..=6 forward (HUD formats it).
    pub fn gear(&self) -> i32 {
        self.cars[self.active].gear()
    }
}

fn car_descriptor_json(car: &Car) -> String {
    let (radius, half_width) = car.wheel_dims();
    let (cmin, cmax) = car.cage();
    let mut s = format!(
        "{{\"wheelRadius\":{},\"wheelHalfWidth\":{},\"wheelColor\":[0.08,0.08,0.10],\
         \"bodyColor\":[0.80,0.16,0.16],\"wheelCount\":{},\
         \"cageMin\":[{},{},{}],\"cageMax\":[{},{},{}],\"chassisCount\":{},\"chassisRest\":[",
        radius, half_width, car.wheel_count(),
        cmin[0], cmin[1], cmin[2], cmax[0], cmax[1], cmax[2], car.chassis_count()
    );
    for (k, v) in car.chassis_rest().iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        s.push_str(&format!("{:.4}", v));
    }
    s.push_str("]}");
    s
}

/// Combined soft-body descriptor for ALL cars: total node count + every beam's
/// endpoint pair using GLOBAL node indices (per-car local index + node offset).
/// The `beams` array only sizes the JS line-index buffer; the actual indices are
/// rewritten each frame from `line_buf` (also global, stress-ordered).
fn soft_descriptor_json(cars: &[Car], offsets: &[usize]) -> String {
    let total_nodes: usize = cars.iter().map(|c| c.node_count()).sum();
    let mut s = format!("{{\"nodeCount\":{},\"color\":[0.85,0.75,0.25],\"beams\":[", total_nodes);
    let mut first = true;
    for (c, car) in cars.iter().enumerate() {
        let off = offsets[c] as u32;
        for v in car.structure.beam_index_pairs() {
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&(v as u32 + off).to_string());
        }
    }
    s.push_str("]}");
    s
}

fn iso_to_mat4(iso: &Isometry<Real>) -> Mat4 {
    let t = iso.translation.vector;
    let r = iso.rotation;
    let q = Quat::from_xyzw(r.i, r.j, r.k, r.w);
    Mat4::from_rotation_translation(q, Vec3::new(t.x, t.y, t.z))
}
