//! Analytic tire force model (brush / Pacejka-lite). Forces are computed at the
//! wheel hub from the slip ratio (longitudinal) and slip angle (lateral), then
//! clamped to the friction circle. The hub is a soft-body node, so suspension
//! and load transfer come from the beam network; this module supplies grip.

/// Saturating slip curve, normalized to [-1, 1]. Rises ~linearly for small slip
/// then saturates — the characteristic tire shape without a full Pacejka fit.
/// `peak` is the slip value at which grip is ~70% of max.
#[inline]
fn slip_curve(slip: f32, peak: f32) -> f32 {
    let t = slip / peak;
    t / (1.0 + t * t).sqrt()
}

/// Longitudinal force from slip ratio `kappa`. Positive `kappa` (wheel spinning
/// faster than ground) drives the car forward.
#[inline]
pub fn longitudinal(fz: f32, mu: f32, kappa: f32) -> f32 {
    mu * fz * slip_curve(kappa, 0.15)
}

/// Lateral force from slip angle `alpha` (radians). Opposes the sideways slip.
#[inline]
pub fn lateral(fz: f32, mu: f32, alpha: f32) -> f32 {
    -mu * fz * slip_curve(alpha, 0.20)
}

/// Clamp the combined (Fx, Fy) to the friction circle `max` so a tire can't
/// exceed total grip (braking + cornering trade off).
#[inline]
pub fn friction_circle(fx: f32, fy: f32, max: f32) -> (f32, f32) {
    let mag = (fx * fx + fy * fy).sqrt();
    if mag > max && mag > 1e-6 {
        let s = max / mag;
        (fx * s, fy * s)
    } else {
        (fx, fy)
    }
}
