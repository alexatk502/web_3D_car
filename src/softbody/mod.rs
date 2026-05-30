//! Soft-body (node/beam) physics — the BeamNG-style core. Phase 0: solver +
//! analytic terrain collision + a test lattice. Driving, drivetrain, skinning
//! and deformation are layered on in later phases.

pub mod car;
pub mod collision;
pub mod drivetrain;
pub mod solver;
pub mod structure;
pub mod tire;

use solver::SolverParams;
use structure::Structure;

/// One deformable body plus its solver settings and a per-frame substep budget.
pub struct SoftBody {
    pub structure: Structure,
    pub params: SolverParams,
}

impl SoftBody {
    /// Phase 0 test body: a soft lattice cube dropped (tilted) above the spawn
    /// point so it lands on an edge and visibly wobbles/deforms.
    pub fn test_lattice() -> Self {
        let spacing = 0.7;
        let dims = [4, 3, 4];
        let origin = [
            -(dims[0] as f32 - 1.0) * spacing * 0.5,
            0.0,
            -(dims[2] as f32 - 1.0) * spacing * 0.5,
        ];
        let mut structure = structure::build_lattice(
            dims, spacing, origin, 2.0, // node mass (kg) — lighter = bouncier
            22_000.0, // beam stiffness — softer = more jiggle
            70.0, // beam damping — underdamped so it oscillates
        );
        // Tilt the whole body and lift it so a corner hits first.
        tilt_and_lift(&mut structure, 0.5, 0.35, crate::scene::terrain_height(0.0, 0.0) + 4.5);
        // Refresh the pristine spawn snapshot so reset (R) re-drops it tilted.
        structure.spawn_px.copy_from_slice(&structure.nodes.px);
        structure.spawn_py.copy_from_slice(&structure.nodes.py);
        structure.spawn_pz.copy_from_slice(&structure.nodes.pz);

        SoftBody {
            structure,
            params: SolverParams::default(),
        }
    }

    /// Run `substeps` solver substeps.
    pub fn run(&mut self, substeps: u32) {
        for _ in 0..substeps {
            solver::substep(&mut self.structure, &self.params);
        }
    }

    pub fn node_count(&self) -> usize {
        self.structure.nodes.len()
    }
}

/// Rotate all nodes about their centroid (Euler X then Z, radians) and lift the
/// body so its lowest node sits at `target_min_y`.
fn tilt_and_lift(s: &mut Structure, rx: f32, rz: f32, target_min_y: f32) {
    let n = &mut s.nodes;
    let count = n.len() as f32;
    let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
    for i in 0..n.len() {
        cx += n.px[i];
        cy += n.py[i];
        cz += n.pz[i];
    }
    cx /= count;
    cy /= count;
    cz /= count;

    let (sx, cxr) = rx.sin_cos();
    let (sz, czr) = rz.sin_cos();
    let mut min_y = f32::INFINITY;
    for i in 0..n.len() {
        let (mut x, mut y, mut z) = (n.px[i] - cx, n.py[i] - cy, n.pz[i] - cz);
        // Rotate about X.
        let (y1, z1) = (y * cxr - z * sx, y * sx + z * cxr);
        y = y1;
        z = z1;
        // Rotate about Z.
        let (x2, y2) = (x * czr - y * sz, x * sz + y * czr);
        x = x2;
        y = y2;
        n.px[i] = cx + x;
        n.py[i] = cy + y;
        n.pz[i] = cz + z;
        min_y = min_y.min(n.py[i]);
    }
    let lift = target_min_y - min_y;
    for i in 0..n.len() {
        n.py[i] += lift;
    }
}
