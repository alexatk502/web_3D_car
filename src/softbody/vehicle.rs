//! Data-driven vehicle specs ("jbeam-lite").
//!
//! The chassis GEOMETRY (grid resolution, cage size, wheel layout, radius) is
//! shared across vehicles so the renderer can reuse one body/tire mesh template.
//! A `VehicleSpec` varies the PHYSICS — weight, stiffness, crumple thresholds,
//! engine torque, grip — and the body colour, giving distinct-handling,
//! distinct-looking vehicle types. Presets are built in; a spec can also be
//! loaded from JSON (partial specs are fine — missing fields fall back to the
//! `sport` defaults), which is the modding hook.

use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct VehicleSpec {
    pub name: String,
    pub body_color: [f32; 3],
    pub chassis_node_mass: f32, // per chassis node (kg); total mass ~ nodes * this
    pub chassis_k: f32,         // axis-beam stiffness (diagonals scale from this)
    pub chassis_deform: f32,    // yield strain (lower = crumples easier)
    pub chassis_break: f32,     // break strain
    pub tire_mu: f32,           // grip
    pub peak_torque: f32,       // engine peak torque (N·m)
}

impl Default for VehicleSpec {
    fn default() -> Self {
        Self::sport()
    }
}

impl VehicleSpec {
    /// The default car (identical to the pre-Phase-9 hardcoded values).
    pub fn sport() -> Self {
        VehicleSpec {
            name: "Sport".into(),
            body_color: [0.80, 0.16, 0.16],
            chassis_node_mass: 8.0,
            chassis_k: 650_000.0,
            chassis_deform: 0.025,
            chassis_break: 0.30,
            tire_mu: 1.7,
            peak_torque: 320.0,
        }
    }

    /// Heavy, soft, lots of crumple — wallows and deforms readily.
    pub fn van() -> Self {
        VehicleSpec {
            name: "Van".into(),
            body_color: [0.20, 0.42, 0.80],
            chassis_node_mass: 13.0,
            chassis_k: 480_000.0,
            chassis_deform: 0.04,
            chassis_break: 0.28,
            tire_mu: 1.45,
            peak_torque: 360.0,
        }
    }

    /// Light, stiff, grippy — zippy and tougher to dent.
    pub fn hatch() -> Self {
        VehicleSpec {
            name: "Hatch".into(),
            body_color: [0.90, 0.78, 0.18],
            chassis_node_mass: 6.0,
            chassis_k: 700_000.0,
            chassis_deform: 0.03,
            chassis_break: 0.32,
            tire_mu: 1.8,
            peak_torque: 240.0,
        }
    }

    /// Built-in presets, indexed by the spawn vehicle id.
    pub fn presets() -> Vec<VehicleSpec> {
        vec![Self::sport(), Self::van(), Self::hatch()]
    }

    /// Parse a (possibly partial) spec from JSON; missing fields use `sport`
    /// defaults. Returns a readable error string on malformed input.
    pub fn from_json(s: &str) -> Result<VehicleSpec, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_full_spec_parses() {
        let s = r#"{"name":"Custom","body_color":[0.1,0.2,0.3],"chassis_node_mass":10.0,
                    "chassis_k":500000.0,"chassis_deform":0.05,"chassis_break":0.4,
                    "tire_mu":1.6,"peak_torque":300.0}"#;
        let v = VehicleSpec::from_json(s).unwrap();
        assert_eq!(v.name, "Custom");
        assert_eq!(v.chassis_node_mass, 10.0);
        assert_eq!(v.peak_torque, 300.0);
    }

    #[test]
    fn json_partial_spec_uses_defaults() {
        // Only override the name + mass; everything else = Sport defaults.
        let v = VehicleSpec::from_json(r#"{"name":"Light","chassis_node_mass":5.0}"#).unwrap();
        assert_eq!(v.name, "Light");
        assert_eq!(v.chassis_node_mass, 5.0);
        assert_eq!(v.chassis_k, VehicleSpec::sport().chassis_k); // defaulted
        assert_eq!(v.tire_mu, VehicleSpec::sport().tire_mu);
    }

    #[test]
    fn json_malformed_errors() {
        assert!(VehicleSpec::from_json("{not valid json").is_err());
    }
}
