// WebGL2 fallback backend. Same Backend interface as the WebGPU one.

import { VERTEX_SRC, FRAGMENT_SRC } from "./shaders.glsl.js";

const LIGHT_DIR = [0.4, 1.0, 0.3];
const IDENTITY = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
// Beam stress heatmap: green → yellow → orange → red.
const STRESS_COLORS = [
  [0.25, 0.80, 0.25],
  [0.90, 0.85, 0.20],
  [0.95, 0.55, 0.10],
  [0.95, 0.15, 0.10],
];

export class WebGLBackend {
  constructor() {
    this.objects = [];
  }

  async init(canvas) {
    const gl = canvas.getContext("webgl2", { antialias: true });
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;
    this.canvas = canvas;

    this.program = linkProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
    this.loc = {
      viewProj: gl.getUniformLocation(this.program, "uViewProj"),
      model: gl.getUniformLocation(this.program, "uModel"),
      color: gl.getUniformLocation(this.program, "uColor"),
      lightDir: gl.getUniformLocation(this.program, "uLightDir"),
    };

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.frontFace(gl.CCW);
    gl.clearColor(0.53, 0.68, 0.86, 1.0);
  }

  uploadMeshes(meshes) {
    const gl = this.gl;
    this.objects = meshes.map((m) => {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);

      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, m.interleaved, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);

      const triBuf = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.triIndices, gl.STATIC_DRAW);

      gl.bindVertexArray(null);

      // Wireframe uses a separate element buffer (not part of the VAO state we
      // rely on; bound explicitly at draw time).
      const lineBuf = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.lineIndices, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

      return {
        vao,
        triBuf,
        lineBuf,
        color: m.color,
        triCount: m.triIndices.length,
        lineCount: m.lineIndices.length,
      };
    });
  }

  // Dynamic soft-body debug mesh (beams as lines), updated each frame.
  setSoftBody({ nodeCount, lineIndices, color }) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, nodeCount * 24, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    const lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
    // Sized to the full beam set; only the unbroken prefix is drawn each frame.
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lineIndices.byteLength, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    this.soft = { vao, vbo, lineBuf, color };
  }

  // Dynamic skinned car body: static triangle topology, per-frame vertices.
  setBody({ maxVerts, triIndices, color }) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, maxVerts * 24, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    const triBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, triIndices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.body = { vao, vbo, color, triCount: triIndices.length };
  }

  setSize(w, h) {
    this.gl.viewport(0, 0, w, h);
  }

  render(viewProj, models, opts) {
    const gl = this.gl;
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.loc.viewProj, false, viewProj);
    gl.uniform3fv(this.loc.lightDir, LIGHT_DIR);

    const wire = opts && opts.wireframe;
    for (let i = 0; i < this.objects.length; i++) {
      const o = this.objects[i];
      gl.uniformMatrix4fv(this.loc.model, false, models[i]);
      gl.uniform3fv(this.loc.color, o.color);
      gl.bindVertexArray(o.vao);
      if (wire) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, o.lineBuf);
        gl.drawElements(gl.LINES, o.lineCount, gl.UNSIGNED_SHORT, 0);
      } else {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, o.triBuf);
        gl.drawElements(gl.TRIANGLES, o.triCount, gl.UNSIGNED_SHORT, 0);
      }
    }

    // Skinned car bodies (solid, no cull). One shared mesh template re-uploaded
    // and drawn once per car from its skinned vertex array.
    if (this.body && opts && opts.body && opts.body.interleavedList) {
      gl.disable(gl.CULL_FACE);
      gl.uniformMatrix4fv(this.loc.model, false, IDENTITY);
      gl.uniform3fv(this.loc.color, this.body.color);
      gl.bindVertexArray(this.body.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.body.vbo);
      for (const interleaved of opts.body.interleavedList) {
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, interleaved);
        gl.drawElements(gl.TRIANGLES, this.body.triCount, gl.UNSIGNED_SHORT, 0);
      }
      gl.enable(gl.CULL_FACE);
    }

    // Soft-body debug lines (always lines, identity model). Vertices follow the
    // nodes; the index buffer holds only unbroken beams.
    if (this.soft && opts && opts.soft && opts.soft.lineCount > 0) {
      gl.uniformMatrix4fv(this.loc.model, false, IDENTITY);
      gl.bindVertexArray(this.soft.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.soft.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, opts.soft.interleaved);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.soft.lineBuf);
      gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, opts.soft.lineIndices);
      const bands = opts.soft.bands;
      if (bands) {
        // Draw each stress band as a contiguous colored segment (heatmap).
        let off = 0;
        for (let bi = 0; bi < 4; bi++) {
          const c = bands[bi];
          if (c > 0) {
            gl.uniform3fv(this.loc.color, STRESS_COLORS[bi]);
            gl.drawElements(gl.LINES, c, gl.UNSIGNED_INT, off * 4); // u32 → byte offset
          }
          off += c;
        }
      } else {
        gl.uniform3fv(this.loc.color, this.soft.color);
        gl.drawElements(gl.LINES, opts.soft.lineCount, gl.UNSIGNED_INT, 0);
      }
    }
    gl.bindVertexArray(null);
  }
}

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("shader compile error: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function linkProgram(gl, vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error("program link error: " + gl.getProgramInfoLog(p));
  }
  return p;
}
