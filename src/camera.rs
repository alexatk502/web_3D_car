//! Camera modes. Produces a right-handed view matrix; the JS side builds the
//! (backend-specific) projection matrix and multiplies the two.

use glam::{Mat4, Quat, Vec3};

#[derive(Clone, Copy, PartialEq)]
pub enum CamMode {
    Chase,
    Hood,
    Orbit,
}

impl CamMode {
    pub fn from_u32(v: u32) -> Self {
        match v % 3 {
            0 => CamMode::Chase,
            1 => CamMode::Hood,
            _ => CamMode::Orbit,
        }
    }
}

pub struct Camera {
    pos: Vec3, // smoothed chase camera position
    orbit_yaw: f32,
    orbit_pitch: f32,
    initialized: bool,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            pos: Vec3::new(0.0, 5.0, -10.0),
            orbit_yaw: 0.0,
            orbit_pitch: 0.4,
            initialized: false,
        }
    }

    /// Compute the view matrix for the current frame.
    ///
    /// `orbit_dx/dy` are pointer-drag deltas (radians) applied only in orbit mode.
    pub fn view(
        &mut self,
        mode: CamMode,
        car_pos: Vec3,
        car_rot: Quat,
        dt: f32,
        orbit_dx: f32,
        orbit_dy: f32,
    ) -> Mat4 {
        let up = Vec3::Y;
        let forward = car_rot * Vec3::X; // chassis forward (+X)

        match mode {
            CamMode::Chase => {
                // Desired position behind + above the car. Use a flattened forward
                // direction so the boom stays level and isn't coupled to car pitch.
                let mut dir = Vec3::new(forward.x, 0.0, forward.z);
                dir = if dir.length_squared() > 1e-4 {
                    dir.normalize()
                } else {
                    forward
                };
                let desired = car_pos - dir * 7.0 + up * 3.5;
                if !self.initialized {
                    self.pos = desired;
                    self.initialized = true;
                } else {
                    let t = (dt * 6.0).min(1.0);
                    self.pos = self.pos.lerp(desired, t);
                }
                Mat4::look_at_rh(self.pos, car_pos + up * 1.0, up)
            }
            CamMode::Hood => {
                let eye = car_pos + car_rot * Vec3::new(0.4, 0.9, 0.0);
                let target = eye + forward * 10.0;
                Mat4::look_at_rh(eye, target, up)
            }
            CamMode::Orbit => {
                self.orbit_yaw += orbit_dx;
                self.orbit_pitch =
                    (self.orbit_pitch + orbit_dy).clamp(-1.4, 1.4);
                let dist = 12.0;
                let (sy, cy) = self.orbit_yaw.sin_cos();
                let (sp, cp) = self.orbit_pitch.sin_cos();
                let offset = Vec3::new(cp * sy, sp, cp * cy) * dist;
                let eye = car_pos + offset;
                Mat4::look_at_rh(eye, car_pos, up)
            }
        }
    }
}
