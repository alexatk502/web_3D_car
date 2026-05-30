// Gouraud-shaded triangles. Lighting is computed in the vertex stage and the
// resulting color is interpolated across the triangle.

struct Frame {
  viewProj : mat4x4<f32>,
  lightDir : vec4<f32>,   // xyz = direction toward the light; w unused
};
@group(0) @binding(0) var<uniform> frame : Frame;

struct Obj {
  model : mat4x4<f32>,
  color : vec4<f32>,      // rgb base color; a unused
};
@group(1) @binding(0) var<uniform> obj : Obj;

struct VsOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) color : vec3<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>, @location(1) normal : vec3<f32>) -> VsOut {
  let world = obj.model * vec4<f32>(position, 1.0);
  // Model matrices are rigid (rotation + translation), so the upper-3x3 of the
  // model matrix transforms normals correctly.
  let n = normalize((obj.model * vec4<f32>(normal, 0.0)).xyz);
  let l = normalize(frame.lightDir.xyz);
  let diff = max(dot(n, l), 0.0);
  let shade = 0.30 + 0.80 * diff;
  var out : VsOut;
  out.clip = frame.viewProj * world;
  out.color = obj.color.rgb * shade;
  return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
