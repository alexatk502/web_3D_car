//! Soft-body data model: struct-of-arrays (SoA) nodes + beams.
//!
//! SoA layout keeps each attribute contiguous, which is cache-friendly and ready
//! for SIMD / parallel iteration in later phases. A node is a point mass; a beam
//! is a spring-damper connecting two nodes. Suspension, body roll, weight
//! transfer and (later) crash deformation all emerge from this network.

/// What role a beam plays. Phase 0 only uses `Normal`; the others are wired in
/// during the vehicle / steering phases.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BeamKind {
    /// Ordinary structural spring.
    Normal,
    /// Stiffer bracing (kept distinct so we can tune classes separately).
    Support,
    /// Input-driven rest length (steering rack, etc.). Rest length is set each
    /// frame from controls.
    Hydro,
}

/// Point masses. `inv_mass == 0` marks a pinned/static node (infinite mass).
#[derive(Default)]
pub struct Nodes {
    pub px: Vec<f32>,
    pub py: Vec<f32>,
    pub pz: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    pub vz: Vec<f32>,
    pub fx: Vec<f32>,
    pub fy: Vec<f32>,
    pub fz: Vec<f32>,
    pub inv_mass: Vec<f32>,
    pub radius: Vec<f32>,
    /// Wheel hub nodes: their ground interaction is handled by the tire model,
    /// so the generic node↔terrain collision skips them.
    pub is_wheel: Vec<bool>,
}

impl Nodes {
    pub fn len(&self) -> usize {
        self.px.len()
    }

    pub fn push(&mut self, pos: [f32; 3], mass: f32, radius: f32) -> u32 {
        let idx = self.px.len() as u32;
        self.px.push(pos[0]);
        self.py.push(pos[1]);
        self.pz.push(pos[2]);
        self.vx.push(0.0);
        self.vy.push(0.0);
        self.vz.push(0.0);
        self.fx.push(0.0);
        self.fy.push(0.0);
        self.fz.push(0.0);
        self.inv_mass.push(if mass > 0.0 { 1.0 / mass } else { 0.0 });
        self.radius.push(radius);
        self.is_wheel.push(false);
        idx
    }

    pub fn mark_wheel(&mut self, i: u32) {
        self.is_wheel[i as usize] = true;
    }

    #[inline]
    pub fn pos(&self, i: usize) -> [f32; 3] {
        [self.px[i], self.py[i], self.pz[i]]
    }
}

/// Spring-dampers between node pairs. `deform`/`break_strain` are consumed by the
/// deformation phase; Phase 0 leaves beams elastic and unbroken.
#[derive(Default)]
pub struct Beams {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub rest: Vec<f32>,
    pub k: Vec<f32>,
    pub d: Vec<f32>,
    pub deform: Vec<f32>,
    pub break_strain: Vec<f32>,
    pub broken: Vec<bool>,
    pub kind: Vec<BeamKind>,
}

impl Beams {
    pub fn len(&self) -> usize {
        self.a.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        a: u32,
        b: u32,
        rest: f32,
        k: f32,
        d: f32,
        deform: f32,
        break_strain: f32,
        kind: BeamKind,
    ) {
        self.a.push(a);
        self.b.push(b);
        self.rest.push(rest);
        self.k.push(k);
        self.d.push(d);
        self.deform.push(deform);
        self.break_strain.push(break_strain);
        self.broken.push(false);
        self.kind.push(kind);
    }
}

/// The full deformable structure: nodes + beams + the pristine rest positions
/// used to respawn on reset.
pub struct Structure {
    pub nodes: Nodes,
    pub beams: Beams,
    pub spawn_px: Vec<f32>,
    pub spawn_py: Vec<f32>,
    pub spawn_pz: Vec<f32>,
    pub spawn_rest: Vec<f32>,
}

impl Structure {
    pub fn new(nodes: Nodes, beams: Beams) -> Self {
        let spawn_px = nodes.px.clone();
        let spawn_py = nodes.py.clone();
        let spawn_pz = nodes.pz.clone();
        let spawn_rest = beams.rest.clone();
        Structure {
            nodes,
            beams,
            spawn_px,
            spawn_py,
            spawn_pz,
            spawn_rest,
        }
    }

    /// Restore the pristine shape and zero all motion (R / reset).
    pub fn reset(&mut self) {
        let n = &mut self.nodes;
        n.px.copy_from_slice(&self.spawn_px);
        n.py.copy_from_slice(&self.spawn_py);
        n.pz.copy_from_slice(&self.spawn_pz);
        for i in 0..n.len() {
            n.vx[i] = 0.0;
            n.vy[i] = 0.0;
            n.vz[i] = 0.0;
        }
        self.beams.rest.copy_from_slice(&self.spawn_rest);
        for b in self.beams.broken.iter_mut() {
            *b = false;
        }
    }

    /// Flat `[x,y,z, ...]` beam endpoint index list for the debug line renderer.
    pub fn beam_index_pairs(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.beams.len() * 2);
        for i in 0..self.beams.len() {
            v.push(self.beams.a[i]);
            v.push(self.beams.b[i]);
        }
        v
    }
}

/// Build a solid lattice box of nodes with structural + shear + bracing beams.
/// `dims` = node counts per axis, `spacing` = node spacing (m), `origin` = center
/// of the bottom face is placed so the box sits with its min corner at `origin`.
pub fn build_lattice(
    dims: [usize; 3],
    spacing: f32,
    origin: [f32; 3],
    node_mass: f32,
    k: f32,
    d: f32,
) -> Structure {
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let mut nodes = Nodes::default();
    let idx = |x: usize, y: usize, z: usize| ((x * ny + y) * nz + z) as u32;

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let pos = [
                    origin[0] + x as f32 * spacing,
                    origin[1] + y as f32 * spacing,
                    origin[2] + z as f32 * spacing,
                ];
                nodes.push(pos, node_mass, spacing * 0.4);
            }
        }
    }

    let mut beams = Beams::default();
    let mut connect = |beams: &mut Beams, na: u32, nb: u32, kk: f32| {
        let a = na as usize;
        let b = nb as usize;
        let dx = nodes.px[a] - nodes.px[b];
        let dy = nodes.py[a] - nodes.py[b];
        let dz = nodes.pz[a] - nodes.pz[b];
        let rest = (dx * dx + dy * dy + dz * dz).sqrt();
        // High plastic/break thresholds in Phase 0 -> effectively elastic.
        beams.push(na, nb, rest, kk, d, 100.0, 100.0, BeamKind::Normal);
    };

    // Connect every node to neighbors within the body-diagonal distance: gives
    // axis edges (structure), face diagonals (shear), and body diagonals (rigidity).
    let max_d2 = (3.0 * spacing * spacing) * 1.05;
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let a = idx(x, y, z);
                // Only look "forward" to avoid duplicate beams.
                for ox in 0..=1usize {
                    for oy in 0..=1usize {
                        for oz in 0..=1usize {
                            if ox == 0 && oy == 0 && oz == 0 {
                                continue;
                            }
                            let (xx, yy, zz) = (x + ox, y + oy, z + oz);
                            if xx >= nx || yy >= ny || zz >= nz {
                                continue;
                            }
                            let b = idx(xx, yy, zz);
                            let d2 = (ox * ox + oy * oy + oz * oz) as f32 * spacing * spacing;
                            if d2 <= max_d2 {
                                // Diagonals get a slightly softer constant.
                                let kk = if ox + oy + oz == 1 { k } else { k * 0.7 };
                                connect(&mut beams, a, b, kk);
                            }
                        }
                    }
                }
            }
        }
    }

    Structure::new(nodes, beams)
}
