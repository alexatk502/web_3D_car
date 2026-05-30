//! Engine + automatic gearbox + final drive. A 1-D rotational model: engine RPM
//! follows the driven wheels through the current gear; throttle scales the torque
//! curve; the auto box shifts on RPM thresholds. (Manual clutch/H-shift is a
//! later polish phase.)

const IDLE_RPM: f32 = 900.0;
const REDLINE_RPM: f32 = 6800.0;
const PEAK_TORQUE: f32 = 320.0; // N·m at the crank
const PEAK_RPM_FRAC: f32 = 0.55; // torque peak at 55% of redline
const FINAL_DRIVE: f32 = 3.7;
const GEARS: [f32; 6] = [3.6, 2.4, 1.7, 1.25, 0.95, 0.78];
const EFFICIENCY: f32 = 0.9;
const ENGINE_BRAKE: f32 = 18.0; // N·m per (rad/s) of engine drag off-throttle
const UPSHIFT_FRAC: f32 = 0.92;
const DOWNSHIFT_FRAC: f32 = 0.38;
const SHIFT_COOLDOWN: f32 = 0.4; // seconds between shifts

const RAD_S_TO_RPM: f32 = 60.0 / (2.0 * std::f32::consts::PI);

pub struct Drivetrain {
    pub gear: usize, // index into GEARS
    pub rpm: f32,
    shift_timer: f32,
}

impl Drivetrain {
    pub fn new() -> Self {
        Drivetrain {
            gear: 0,
            rpm: IDLE_RPM,
            shift_timer: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.gear = 0;
        self.rpm = IDLE_RPM;
        self.shift_timer = 0.0;
    }

    /// Normalized engine torque curve (0..1) vs rpm — rises to a mid-range peak,
    /// tapers toward redline, near zero past it.
    fn torque_factor(rpm: f32) -> f32 {
        let peak = REDLINE_RPM * PEAK_RPM_FRAC;
        let t = (rpm - peak) / peak;
        (1.0 - 0.7 * t * t).clamp(0.15, 1.0)
    }

    /// Advance the driveline and return the **total drive torque delivered at the
    /// wheels** (to be split across driven wheels). `wheel_omega` is the average
    /// driven-wheel angular speed (rad/s). Engine braking is folded in when
    /// `throttle` is low.
    pub fn update(&mut self, throttle: f32, wheel_omega: f32, dt: f32) -> f32 {
        let ratio = GEARS[self.gear] * FINAL_DRIVE;

        // Engine speed follows the wheels through the gearing (auto clutch lock).
        let engine_omega = (wheel_omega.max(0.0)) * ratio;
        self.rpm = (engine_omega * RAD_S_TO_RPM).clamp(IDLE_RPM, REDLINE_RPM + 200.0);

        // Auto shifting.
        self.shift_timer = (self.shift_timer - dt).max(0.0);
        if self.shift_timer == 0.0 {
            if self.rpm > REDLINE_RPM * UPSHIFT_FRAC && self.gear < GEARS.len() - 1 {
                self.gear += 1;
                self.shift_timer = SHIFT_COOLDOWN;
            } else if self.rpm < REDLINE_RPM * DOWNSHIFT_FRAC && self.gear > 0 {
                self.gear -= 1;
                self.shift_timer = SHIFT_COOLDOWN;
            }
        }

        let crank_torque = throttle * PEAK_TORQUE * Self::torque_factor(self.rpm);
        let drive = crank_torque * ratio * EFFICIENCY;

        // Engine braking when off-throttle (drag scaled by engine speed).
        let brake = if throttle < 0.05 {
            ENGINE_BRAKE * engine_omega * (1.0 - throttle) / ratio.max(0.1)
        } else {
            0.0
        };

        drive - brake
    }
}
