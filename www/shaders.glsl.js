// GLSL ES 3.00 shaders for the WebGL2 backend. Mirrors shader.wgsl: per-vertex
// (Gouraud) lighting, color interpolated to the fragment stage.

export const VERTEX_SRC = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uViewProj;
uniform mat4 uModel;
uniform vec3 uColor;
uniform vec3 uLightDir;

out vec3 vColor;

void main() {
  vec4 world = uModel * vec4(aPosition, 1.0);
  vec3 n = normalize((uModel * vec4(aNormal, 0.0)).xyz);
  vec3 l = normalize(uLightDir);
  float diff = max(dot(n, l), 0.0);
  float shade = 0.30 + 0.80 * diff;
  vColor = uColor * shade;
  gl_Position = uViewProj * world;
}`;

export const FRAGMENT_SRC = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 outColor;
void main() {
  outColor = vec4(vColor, 1.0);
}`;
