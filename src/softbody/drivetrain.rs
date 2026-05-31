//! Engine + clutch + gearbox + final drive — a 1-D rotational driveline.
//!
//! The engine is a free rotational **inertia**: throttle and an idle governor
//! spin it up, internal drag spins it down (engine braking). A **clutch**
//! transmits torque between the engine and the gearbox input, *slipping* when
//! their speeds differ (so you can rev in neutral, launch from a stop, or stall
//! the revs under load). Two modes: **Auto** (auto-shift + auto-clutch +
//! brake-to-reverse) and **Manual** (player clutch pedal + sequential H-shift
//! through R-N-1..6). Returns the signed drive torque delivered at the wheels.

const IDLE_RPM: f32 = 900.0;
const REDLINE_RPM: f32 = 6800.0;
const PEAK_TORQUE: f32 = 320.0; // N·m at the crank
const PEAK_RPM_FRAC: f32 = 0.55; // torque peak at 55% of redline
const FINAL_DRIVE: f32 = 3.7;
const GEARS: [f32; 6] = [3.6, 2.4, 1.7, 1.25, 0.95, 0.78];
const REVERSE_RATIO: f32 = 3.9;
const EFFICIENCY: f32 = 0.9;

const ENGINE_INERTIA: f32 = 0.22; // kg·m² (engine + flywheel) — snappy but smooth
const ENGINE_DRAG: f32 = 0.045; // N·m per rad/s — internal friction = engine braking
const IDLE_TORQUE: f32 = 60.0; // idle governor strength (holds ~IDLE_RPM)
const CLUTCH_MAX: f32 = 440.0; // max transmissible clutch torque (N·m)
const CLUTCH_STIFFNESS: f32 = 6.0; // N·m per rad/s of clutch slip

const UPSHIFT_FRAC: f32 = 0.92;
const DOWNSHIFT_FRAC: f32 = 0.40;
const SHIFT_COOLDOWN: f32 = 0.4; // seconds between auto shifts
const SHIFT_DECLUTCH: f32 = 0.15; // clutch-slip window during a shift

const RAD_S_TO_RPM: f32 = 60.0 / (2.0 * std::f32::consts::PI);

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Auto,
    Manual,
}

pub struct Drivetrain {
    pub gear: i32, // -1 = reverse, 0 = neutral, 1..=6 forward
    pub rpm: f32,
    pub mode: Mode,
    pub clutch: f32,   // current engagement 0..1 (1 = locked), for the HUD
    engine_omega: f32, // rad/s
    shift_timer: f32,
    declutch_timer: f32,
}

impl Drivetrain {
    pub fn new() -> Self {
        Drivetrain {
            gear: 1,
            rpm: IDLE_RPM,
            mode: Mode::Auto,
            clutch: 1.0,
            engine_omega: IDLE_RPM / RAD_S_TO_RPM,
            shift_timer: 0.0,
            declutch_timer: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.gear = if self.mode == Mode::Manual { 0 } else { 1 };
        self.rpm = IDLE_RPM;
        self.clutch = 1.0;
        self.engine_omega = IDLE_RPM / RAD_S_TO_RPM;
        self.shift_timer = 0.0;
        self.declutch_timer = 0.0;
    }

    pub fn set_manual(&mut self, on: bool) {
        let want = if on { Mode::Manual } else { Mode::Auto };
        if want != self.mode {
            self.mode = want;
            // Entering auto: make sure we're in a drive gear. Entering manual:
            // drop to neutral so the player consciously selects a gear.
            self.gear = if on { 0 } else { 1 };
        }
    }

    pub fn toggle_manual(&mut self) {
        self.set_manual(self.mode == Mode::Auto);
    }

    /// Sequential upshift (R -> N -> 1 -> .. -> 6). Manual mode.
    pub fn shift_up(&mut self) {
        if self.mode == Mode::Manual && self.gear < 6 {
            self.gear += 1;
            self.declutch_timer = self.declutch_timer.max(0.04);
        }
    }
    /// Sequential downshift (6 -> .. -> 1 -> N -> R). Manual mode.
    pub fn shift_down(&mut self) {
        if self.mode == Mode::Manual && self.gear > -1 {
            self.gear -= 1;
            self.declutch_timer = self.declutch_timer.max(0.04);
        }
    }

    /// True when auto mode is driving in reverse (brake pedal acts as reverse
    /// throttle, so the friction brakes should be suppressed).
    pub fn auto_reversing(&self) -> bool {
        self.mode == Mode::Auto && self.gear == -1
    }

    fn ratio(&self) -> f32 {
        match self.gear {
            -1 => -REVERSE_RATIO * FINAL_DRIVE,
            0 => 0.0,
            g => GEARS[(g - 1) as usize] * FINAL_DRIVE,
        }
    }

    /// Normalized engine torque curve (0..1) vs rpm.
    fn torque_factor(rpm: f32) -> f32 {
        let peak = REDLINE_RPM * PEAK_RPM_FRAC;
        let t = (rpm - peak) / peak;
        (1.0 - 0.7 * t * t).clamp(0.15, 1.0)
    }

    /// Advance the driveline one substep and return the signed drive torque at the
    /// wheels (negative = reverse). `wheel_omega` is the average driven-wheel
    /// angular speed (rad/s, signed). `clutch_pedal` is 0 (engaged) .. 1
    /// (disengaged) — only used in manual mode.
    pub fn update(
        &mut self,
        throttle: f32,
        brake: f32,
        clutch_pedal: f32,
        wheel_omega: f32,
        dt: f32,
    ) -> f32 {
        self.shift_timer = (self.shift_timer - dt).max(0.0);
        self.declutch_timer = (self.declutch_timer - dt).max(0.0);

        let mut eff_throttle = throttle.clamp(0.0, 1.0);
        let engagement;

        match self.mode {
            Mode::Manual => {
                let pedal = (1.0 - clutch_pedal).clamp(0.0, 1.0);
                engagement = if self.declutch_timer > 0.0 { 0.0 } else { pedal };
            }
            Mode::Auto => {
                // Brake-to-reverse at a standstill; throttle returns to drive.
                if self.gear >= 0 && throttle < 0.05 && brake > 0.05 && wheel_omega < 1.0 {
                    self.gear = -1;
                } else if self.gear == -1 && throttle > 0.05 {
                    self.gear = 1;
                }
                if self.gear == -1 {
                    eff_throttle = brake.clamp(0.0, 1.0); // brake pedal drives reverse
                } else if self.gear == 0 {
                    self.gear = 1; // auto never idles in neutral
                }
                // Auto shifting on rpm thresholds (forward gears only).
                if self.gear >= 1 && self.shift_timer == 0.0 {
                    if self.rpm > REDLINE_RPM * UPSHIFT_FRAC && self.gear < 6 {
                        self.gear += 1;
                        self.shift_timer = SHIFT_COOLDOWN;
                        self.declutch_timer = SHIFT_DECLUTCH;
                    } else if self.rpm < REDLINE_RPM * DOWNSHIFT_FRAC && self.gear > 1 {
                        self.gear -= 1;
                        self.shift_timer = SHIFT_COOLDOWN;
                        self.declutch_timer = SHIFT_DECLUTCH;
                    }
                }
                engagement = if self.declutch_timer > 0.0 { 0.0 } else { 1.0 };
            }
        }
        self.clutch = engagement;

        let ratio = self.ratio();

        // Clutch torque between the engine and the gearbox input. Positive torque
        // (engine spinning faster than the geared-up wheels) drives the wheels;
        // negative (wheels back-driving a slower engine) is engine braking.
        let mut t_clutch = if ratio != 0.0 && engagement > 0.0 {
            let gbox_omega = wheel_omega * ratio;
            let slip = self.engine_omega - gbox_omega;
            (CLUTCH_STIFFNESS * slip).clamp(-CLUTCH_MAX, CLUTCH_MAX) * engagement
        } else {
            0.0
        };
        // Auto + off-throttle: allow only braking torque, never drive — so the idle
        // engine doesn't creep the car forward when stopped, while engine braking
        // (negative torque, wheels back-driving the engine) still works.
        if self.mode == Mode::Auto && eff_throttle < 0.05 {
            t_clutch = t_clutch.min(0.0);
        }

        // Engine dynamics: throttle torque + idle governor − drag − clutch load.
        let t_throttle = eff_throttle * PEAK_TORQUE * Self::torque_factor(self.rpm);
        let t_idle = if self.rpm < IDLE_RPM * 1.15 {
            IDLE_TORQUE * ((IDLE_RPM - self.rpm) / IDLE_RPM).max(0.0)
        } else {
            0.0
        };
        let t_drag = ENGINE_DRAG * self.engine_omega;
        let net = t_throttle + t_idle - t_drag - t_clutch;
        self.engine_omega = (self.engine_omega + net / ENGINE_INERTIA * dt).max(0.0);

        // Auto mode never bogs below idle (a torque-converter-like floor), so it
        // can't stall at a clutch-grabbing launch. Manual lets the revs drop.
        if self.mode == Mode::Auto {
            self.engine_omega = self.engine_omega.max(IDLE_RPM / RAD_S_TO_RPM);
        }

        let max_omega = (REDLINE_RPM + 300.0) / RAD_S_TO_RPM;
        if self.engine_omega > max_omega {
            self.engine_omega = max_omega;
        }
        self.rpm = (self.engine_omega * RAD_S_TO_RPM).clamp(0.0, REDLINE_RPM + 300.0);

        // Torque at the wheels = clutch torque geared up (signed by the ratio).
        t_clutch * ratio * EFFICIENCY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 2000.0;

    fn run(dt: &mut Drivetrain, throttle: f32, brake: f32, clutch: f32, wheel_omega: f32, n: usize) -> f32 {
        let mut last = 0.0;
        for _ in 0..n {
            last = dt.update(throttle, brake, clutch, wheel_omega, DT);
        }
        last
    }

    #[test]
    fn auto_full_throttle_drives_forward() {
        let mut d = Drivetrain::new(); // Auto, gear 1
        // Wheels barely turning, full throttle: clutch should transmit forward torque.
        let tq = run(&mut d, 1.0, 0.0, 0.0, 2.0, 200);
        assert!(tq > 0.0, "auto full throttle should drive forward, got {}", tq);
        // Auto holds at least idle even when the clutch loads it down.
        assert!(d.rpm >= IDLE_RPM * 0.99, "auto should not bog below idle, rpm={}", d.rpm);
    }

    #[test]
    fn clutch_disengaged_transmits_no_drive_but_engine_revs() {
        let mut d = Drivetrain::new();
        d.set_manual(true); // drops to neutral
        d.shift_up(); // N -> 1
        assert_eq!(d.gear, 1);
        // Clutch fully disengaged (pedal=1): wheels get ~no torque, engine free-revs.
        let tq = run(&mut d, 1.0, 0.0, 1.0, 1.0, 1000);
        assert!(tq.abs() < 1.0, "disengaged clutch should transmit ~no torque, got {}", tq);
        assert!(d.rpm > 1500.0, "engine should rev freely with clutch in, rpm={}", d.rpm);
    }

    #[test]
    fn manual_reverse_gives_negative_wheel_torque() {
        let mut d = Drivetrain::new();
        d.set_manual(true); // neutral
        d.shift_down(); // N -> R
        assert_eq!(d.gear, -1);
        let tq = run(&mut d, 1.0, 0.0, 0.0, -1.0, 200); // throttle, clutch engaged
        assert!(tq < 0.0, "reverse gear should drive wheels backward, got {}", tq);
    }

    #[test]
    fn neutral_revs_free_no_drive() {
        let mut d = Drivetrain::new();
        d.set_manual(true); // neutral (gear 0)
        assert_eq!(d.gear, 0);
        let tq = run(&mut d, 1.0, 0.0, 0.0, 0.0, 1000);
        assert!(tq.abs() < 1e-3, "neutral transmits no torque, got {}", tq);
        assert!(d.rpm > 1500.0, "neutral full throttle should rev out, rpm={}", d.rpm);
    }

    #[test]
    fn off_throttle_engine_braking_is_negative() {
        let mut d = Drivetrain::new(); // Auto, gear 1
        // Spin engine up first, then coast at speed with no throttle: clutch should
        // back-drive the (slower) engine, i.e. deliver negative (braking) torque.
        run(&mut d, 1.0, 0.0, 0.0, 60.0, 200);
        let tq = d.update(0.0, 0.0, 0.0, 60.0, DT);
        assert!(tq < 0.0, "closed throttle at speed should give engine braking, got {}", tq);
    }
}
