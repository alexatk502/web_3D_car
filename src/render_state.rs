//! Flat `f32` buffer shared with JS via WASM linear memory (zero-copy).
//!
//! Layout (column-major mat4s, 16 floats each):
//!   [ view ][ model_0 ][ model_1 ] ... [ model_{n-1} ]
//! The model order matches the scene descriptor order returned by `World`.

use glam::Mat4;

pub struct RenderBuffer {
    pub data: Vec<f32>,
}

impl RenderBuffer {
    pub fn new(num_objects: usize) -> Self {
        Self {
            data: vec![0.0; 16 * (num_objects + 1)],
        }
    }

    pub fn set_view(&mut self, m: &Mat4) {
        self.data[0..16].copy_from_slice(&m.to_cols_array());
    }

    /// `idx` is the object index (0-based); model 0 lives right after the view.
    pub fn set_model(&mut self, idx: usize, m: &Mat4) {
        let o = 16 * (idx + 1);
        self.data[o..o + 16].copy_from_slice(&m.to_cols_array());
    }

    pub fn ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}
