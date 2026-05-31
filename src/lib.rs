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
    car: Car,
    camera: Camera,
    static_handles: Vec<RigidBodyHandle>,
    descriptor: String,
    soft_descriptor: String,
    car_descriptor: String,
    num_static: usize,  // terrain + obstacles
    num_objects: usize, // num_static + wheels (total model matrices in buffer)
    buffer: RenderBuffer,
    node_buf: Vec<f32>, // [x,y,z, ...] per node, refreshed each frame
    line_buf: Vec<u16>, // active (unbroken) beam endpoint pairs, refreshed each frame
    line_count: usize,  // number of valid indices in line_buf this frame
    input: Input,
    accumulator: f32,
    last_substeps: u32, // substeps run on the most recent frame (HUD)
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
        let soft_descriptor = soft_descriptor_json(&car);
        let car_descriptor = car_descriptor_json(&car);
        // Render buffer holds static models + one model per wheel.
        let num_objects = num_static + car.wheel_count();
        let node_buf = vec![0.0; car.node_count() * 3];
        let line_buf = vec![0u16; car.structure.beams.len() * 2];

        World {
            physics,
            car,
            camera: Camera::new(),
            static_handles,
            descriptor,
            soft_descriptor,
            car_descriptor,
            num_static,
            num_objects,
            buffer: RenderBuffer::new(num_objects),
            node_buf,
            line_buf,
            line_count: 0,
            input: Input::default(),
            accumulator: 0.0,
            last_substeps: 0,
        }
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

    /// Number of wheels (model matrices that follow the static objects).
    pub fn wheel_count(&self) -> usize {
        self.car.wheel_count()
    }

    /// Enable the parallel (multi-threaded) solver path. Call from JS only after
    /// `initThreadPool` has resolved, so the page is cross-origin isolated and the
    /// rayon worker pool exists.
    pub fn set_threaded(&mut self, on: bool) {
        self.car.set_threaded(on);
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

    /// Manual gearbox: sequential shift up/down (R-N-1..6) and auto/manual toggle.
    pub fn shift_up(&mut self) {
        self.car.shift_up();
    }
    pub fn shift_down(&mut self) {
        self.car.shift_down();
    }
    pub fn toggle_manual(&mut self) {
        self.car.toggle_manual();
    }
    pub fn is_manual(&self) -> bool {
        self.car.is_manual()
    }
    /// Clutch engagement 0..1 (1 = locked), for the HUD.
    pub fn clutch(&self) -> f32 {
        self.car.clutch_engagement()
    }

    /// Advance by `dt` real seconds (substep accumulator), then recompute the
    /// camera and refill the render + node buffers.
    pub fn step(&mut self, dt: f32, camera_mode: u32, orbit_dx: f32, orbit_dy: f32) {
        // Push the current input into the car (reset is handled inside set_input).
        self.car.set_input(
            self.input.throttle,
            self.input.brake,
            self.input.steer,
            self.input.handbrake,
            self.input.reset,
            self.input.clutch,
        );

        let substep_dt = self.car.params.substep_dt;
        self.accumulator += dt.min(0.1);
        let mut steps = 0u32;
        while self.accumulator >= substep_dt && steps < MAX_SUBSTEPS_PER_FRAME {
            self.car.run(1);
            self.accumulator -= substep_dt;
            steps += 1;
        }
        // Drop any leftover backlog so we don't spiral after a stall.
        if self.accumulator > substep_dt {
            self.accumulator = 0.0;
        }
        self.last_substeps = steps;

        self.refill_buffer(camera_mode, dt, orbit_dx, orbit_dy);
        self.refill_nodes();
        self.refill_lines();
    }

    fn refill_buffer(&mut self, camera_mode: u32, dt: f32, orbit_dx: f32, orbit_dy: f32) {
        // Camera tracks the car centroid, oriented to its heading.
        let target = self.car.centroid();
        let fwd = self.car.forward();
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

        // Static objects (terrain + obstacles) by their fixed body transforms.
        for (idx, h) in self.static_handles.iter().enumerate() {
            let iso = self.physics.bodies[*h].position();
            self.buffer.set_model(idx, &iso_to_mat4(iso));
        }
        // Wheels: model matrices follow the hub nodes (+ steer + spin).
        for i in 0..self.car.wheel_count() {
            let (pos, rot) = self.car.wheel_transform(i);
            self.buffer
                .set_model(self.num_static + i, &Mat4::from_rotation_translation(rot, pos));
        }
    }

    fn refill_nodes(&mut self) {
        let n = &self.car.structure.nodes;
        for i in 0..n.len() {
            self.node_buf[i * 3] = n.px[i];
            self.node_buf[i * 3 + 1] = n.py[i];
            self.node_buf[i * 3 + 2] = n.pz[i];
        }
    }

    /// Fill `line_buf` with the endpoint pairs of unbroken beams only, so broken
    /// beams vanish from the debug renderer.
    fn refill_lines(&mut self) {
        let b = &self.car.structure.beams;
        let mut k = 0;
        for i in 0..b.len() {
            if !b.broken[i] {
                self.line_buf[k] = b.a[i] as u16;
                self.line_buf[k + 1] = b.b[i] as u16;
                k += 2;
            }
        }
        self.line_count = k;
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
    pub fn node_count(&self) -> usize {
        self.car.node_count()
    }

    /// Total beam count (including broken), for the HUD.
    pub fn beam_count(&self) -> usize {
        self.car.structure.beams.len()
    }

    /// Substeps executed on the most recent frame (HUD).
    pub fn substeps_last_frame(&self) -> u32 {
        self.last_substeps
    }

    // --- Active (unbroken) beam line indices (for the debug renderer) ---
    pub fn line_buffer_ptr(&self) -> *const u16 {
        self.line_buf.as_ptr()
    }
    pub fn line_buffer_len(&self) -> usize {
        self.line_buf.len()
    }
    /// Number of valid indices in the line buffer this frame (2 per unbroken beam).
    pub fn line_count(&self) -> usize {
        self.line_count
    }

    /// Car forward speed in km/h.
    pub fn speed_kmh(&self) -> f32 {
        self.car.speed_kmh()
    }

    /// Engine RPM (HUD).
    pub fn rpm(&self) -> f32 {
        self.car.rpm()
    }

    /// Current gear: -1 = reverse, 0 = neutral, 1..=6 forward (HUD formats it).
    pub fn gear(&self) -> i32 {
        self.car.gear()
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

fn soft_descriptor_json(car: &Car) -> String {
    let beams = car.structure.beam_index_pairs();
    let mut s = format!("{{\"nodeCount\":{},\"color\":[0.85,0.75,0.25],\"beams\":[", car.node_count());
    for (k, v) in beams.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
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
