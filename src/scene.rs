//! Builds the static world (procedural rocky terrain + scattered obstacles) and
//! records a render descriptor so the JS side builds geometry whose dimensions
//! exactly match the Rapier colliders. The descriptor is the single source of
//! truth for object sizes/colors.
//!
//! The terrain mesh is generated once here and used for BOTH the Rapier trimesh
//! collider and the rendered geometry, so collision always matches the visuals.

use crate::physics::Physics;
use rapier3d::prelude::*;

// --- Terrain parameters ---
const TERRAIN_SPAN: f32 = 220.0; // world size along X and Z (centered on origin)
const TERRAIN_CELLS: usize = 64; // grid resolution (cells per side)
const FLAT_RADIUS: f32 = 14.0; // flat spawn area radius
const FLAT_BLEND: f32 = 14.0; // blend distance from flat area into rocky terrain

/// What kind of mesh a renderable object uses.
pub enum MeshKind {
    /// A box with the given half-extents.
    Box { hx: f32, hy: f32, hz: f32 },
    /// A wheel cylinder: `radius` in the local XY plane, axis along local Z.
    Cylinder { radius: f32, half_width: f32 },
    /// A triangle mesh (terrain). Flat arrays: `vertices` = [x,y,z,...],
    /// `indices` = [i0,i1,i2,...]. JS computes per-vertex normals.
    Terrain { vertices: Vec<f32>, indices: Vec<u32> },
}

/// One renderable object's static description (geometry + color).
pub struct ObjDesc {
    pub kind: MeshKind,
    pub color: [f32; 3],
}

// --- Procedural height field (deterministic value noise) ---

fn hash2(i: i32, j: i32) -> f32 {
    let mut h = (i.wrapping_mul(374_761_393)).wrapping_add(j.wrapping_mul(668_265_263)) as u32;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0 // -1..1
}

fn smoother(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn value_noise(x: f32, z: f32, freq: f32) -> f32 {
    let (xs, zs) = (x * freq, z * freq);
    let (x0, z0) = (xs.floor() as i32, zs.floor() as i32);
    let (fx, fz) = (xs - x0 as f32, zs - z0 as f32);
    let (u, v) = (smoother(fx), smoother(fz));
    let n00 = hash2(x0, z0);
    let n10 = hash2(x0 + 1, z0);
    let n01 = hash2(x0, z0 + 1);
    let n11 = hash2(x0 + 1, z0 + 1);
    let nx0 = n00 + (n10 - n00) * u;
    let nx1 = n01 + (n11 - n01) * u;
    nx0 + (nx1 - nx0) * v
}

/// When true the ground is perfectly flat (y = 0); set false to restore the
/// procedural rocky terrain below.
const FLAT_GROUND: bool = true;

/// Terrain height at world (x, z). Flat near the origin (spawn), rocky beyond.
pub fn terrain_height(x: f32, z: f32) -> f32 {
    if FLAT_GROUND {
        return 0.0;
    }
    let base = value_noise(x, z, 0.030) * 4.5  // broad rolling hills
        + value_noise(x, z, 0.090) * 1.6        // medium bumps
        + value_noise(x, z, 0.240) * 0.55;      // rocky detail
    // Flatten toward the origin so the car spawns on level ground.
    let d = (x * x + z * z).sqrt();
    let t = ((d - FLAT_RADIUS) / FLAT_BLEND).clamp(0.0, 1.0);
    base * smoother(t)
}

/// Unit surface normal of the terrain at (x, z), from the analytic gradient.
/// Used by soft-body node↔terrain collision.
pub fn terrain_normal(x: f32, z: f32) -> [f32; 3] {
    let e = 0.5; // finite-difference step
    let dhdx = (terrain_height(x + e, z) - terrain_height(x - e, z)) / (2.0 * e);
    let dhdz = (terrain_height(x, z + e) - terrain_height(x, z - e)) / (2.0 * e);
    // Surface n = normalize(-dh/dx, 1, -dh/dz).
    let (nx, ny, nz) = (-dhdx, 1.0, -dhdz);
    let inv = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
    [nx * inv, ny * inv, nz * inv]
}

fn generate_terrain() -> (Vec<Point<Real>>, Vec<[u32; 3]>) {
    let cells = TERRAIN_CELLS;
    let n = cells + 1;
    let step = TERRAIN_SPAN / cells as f32;
    let half = TERRAIN_SPAN * 0.5;

    let mut verts = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let x = -half + j as f32 * step;
            let z = -half + i as f32 * step;
            verts.push(point![x, terrain_height(x, z), z]);
        }
    }

    let mut tris = Vec::with_capacity(cells * cells * 2);
    for i in 0..cells {
        for j in 0..cells {
            let a = (i * n + j) as u32;
            let b = (i * n + j + 1) as u32;
            let c = ((i + 1) * n + j) as u32;
            let d = ((i + 1) * n + j + 1) as u32;
            // Wound so the geometric normal points +Y (up), matching back-face cull.
            tris.push([a, c, b]);
            tris.push([b, c, d]);
        }
    }
    (verts, tris)
}

/// Serialize the descriptor list to a compact JSON array string (no serde dep).
pub fn descriptor_json(objs: &[ObjDesc]) -> String {
    let mut s = String::from("[");
    for (i, o) in objs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        match &o.kind {
            MeshKind::Box { hx, hy, hz } => {
                s.push_str(&format!(
                    "{{\"kind\":\"box\",\"hx\":{},\"hy\":{},\"hz\":{},\"color\":[{},{},{}]}}",
                    hx, hy, hz, o.color[0], o.color[1], o.color[2]
                ));
            }
            MeshKind::Cylinder { radius, half_width } => {
                s.push_str(&format!(
                    "{{\"kind\":\"cylinder\",\"radius\":{},\"halfWidth\":{},\"color\":[{},{},{}]}}",
                    radius, half_width, o.color[0], o.color[1], o.color[2]
                ));
            }
            MeshKind::Terrain { vertices, indices } => {
                s.push_str(&format!(
                    "{{\"kind\":\"terrain\",\"color\":[{},{},{}],\"vertices\":[",
                    o.color[0], o.color[1], o.color[2]
                ));
                for (k, v) in vertices.iter().enumerate() {
                    if k > 0 {
                        s.push(',');
                    }
                    s.push_str(&format!("{:.3}", v));
                }
                s.push_str("],\"indices\":[");
                for (k, v) in indices.iter().enumerate() {
                    if k > 0 {
                        s.push(',');
                    }
                    s.push_str(&v.to_string());
                }
                s.push_str("]}");
            }
        }
    }
    s.push(']');
    s
}

/// Spawn the rocky terrain and a handful of fixed obstacle boxes resting on it.
/// Returns the fixed-body handles (in render order) and their descriptors.
pub fn build_static(physics: &mut Physics) -> (Vec<RigidBodyHandle>, Vec<ObjDesc>) {
    let mut handles = Vec::new();
    let mut descs = Vec::new();

    // --- Terrain: one mesh used for both the collider and the render geometry.
    let (verts, tris) = generate_terrain();
    let terrain_body = physics.bodies.insert(RigidBodyBuilder::fixed());
    physics.colliders.insert_with_parent(
        ColliderBuilder::trimesh(verts.clone(), tris.clone()).friction(1.0),
        terrain_body,
        &mut physics.bodies,
    );
    handles.push(terrain_body);

    let mut flat_verts = Vec::with_capacity(verts.len() * 3);
    for p in &verts {
        flat_verts.push(p.x);
        flat_verts.push(p.y);
        flat_verts.push(p.z);
    }
    let mut flat_idx = Vec::with_capacity(tris.len() * 3);
    for t in &tris {
        flat_idx.extend_from_slice(t);
    }
    descs.push(ObjDesc {
        kind: MeshKind::Terrain {
            vertices: flat_verts,
            indices: flat_idx,
        },
        color: [0.42, 0.40, 0.36],
    });

    // --- Obstacles: deterministic boxes resting on the terrain surface.
    for o in obstacle_boxes() {
        let body = physics
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(vector![o.center[0], o.center[1], o.center[2]]));
        physics.colliders.insert_with_parent(
            ColliderBuilder::cuboid(o.half[0], o.half[1], o.half[2]).friction(0.8),
            body,
            &mut physics.bodies,
        );
        handles.push(body);
        descs.push(ObjDesc {
            kind: MeshKind::Box {
                hx: o.half[0],
                hy: o.half[1],
                hz: o.half[2],
            },
            color: o.color,
        });
    }

    (handles, descs)
}

/// An axis-aligned obstacle box (resting on the terrain). Shared by the renderer
/// (via `build_static`) and soft-body collision so they always match.
pub struct ObstacleBox {
    pub center: [f32; 3],
    pub half: [f32; 3],
    pub color: [f32; 3],
}

/// Deterministic obstacle layout. Y is set so each box rests on the terrain.
pub fn obstacle_boxes() -> Vec<ObstacleBox> {
    // (x, z, hx, hy, hz, color)
    let defs: [(f32, f32, f32, f32, f32, [f32; 3]); 10] = [
        (10.0, 6.0, 1.0, 1.0, 1.0, [0.85, 0.45, 0.15]),
        (-8.0, 12.0, 1.5, 0.8, 1.5, [0.20, 0.45, 0.85]),
        (20.0, -12.0, 0.8, 1.5, 0.8, [0.80, 0.20, 0.55]),
        (-22.0, -6.0, 2.0, 0.6, 1.0, [0.55, 0.75, 0.25]),
        (4.0, 24.0, 1.0, 2.0, 1.0, [0.85, 0.75, 0.20]),
        (-26.0, 20.0, 1.2, 1.0, 1.2, [0.30, 0.70, 0.70]),
        (28.0, 12.0, 0.7, 0.7, 3.0, [0.65, 0.30, 0.80]),
        (0.0, -22.0, 3.0, 0.5, 0.7, [0.90, 0.55, 0.40]),
        (-10.0, -28.0, 1.0, 1.2, 1.0, [0.55, 0.52, 0.50]),
        (24.0, 30.0, 1.4, 0.9, 1.4, [0.55, 0.52, 0.50]),
    ];
    defs.iter()
        .map(|&(x, z, hx, hy, hz, color)| ObstacleBox {
            center: [x, terrain_height(x, z) + hy, z],
            half: [hx, hy, hz],
            color,
        })
        .collect()
}
