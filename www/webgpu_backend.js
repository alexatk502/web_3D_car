// WebGPU rendering backend. Shares the Backend interface with the WebGL2 one:
//   init(canvas), uploadMeshes(meshes), setSize(w,h), render(viewProj, models, opts)

const LIGHT_DIR = [0.4, 1.0, 0.3];

export class WebGPUBackend {
  constructor() {
    this.device = null;
    this.objects = [];
    this.depthTexture = null;
  }

  async init(canvas) {
    if (!navigator.gpu) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    this.device = await adapter.requestDevice();
    this.canvas = canvas;
    this.context = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque",
    });

    const src = await (await fetch("./shader.wgsl")).text();
    const module = this.device.createShaderModule({ code: src });

    // group 0: per-frame (viewProj + light); group 1: per-object (model + color).
    this.frameBGL = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }],
    });
    this.objBGL = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }],
    });
    const layout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.objBGL],
    });

    const vertexState = {
      module,
      entryPoint: "vs",
      buffers: [
        {
          arrayStride: 24, // 6 floats
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        },
      ],
    };
    const fragmentState = {
      module,
      entryPoint: "fs",
      targets: [{ format: this.format }],
    };
    const depthStencil = {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    };

    this.solidPipeline = this.device.createRenderPipeline({
      layout,
      vertex: vertexState,
      fragment: fragmentState,
      primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
      depthStencil,
    });
    this.wirePipeline = this.device.createRenderPipeline({
      layout,
      vertex: vertexState,
      fragment: fragmentState,
      primitive: { topology: "line-list" },
      depthStencil,
    });
    // Body mesh: solid, but no back-face culling (FFD can flip winding).
    this.bodyPipeline = this.device.createRenderPipeline({
      layout,
      vertex: vertexState,
      fragment: fragmentState,
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil,
    });

    // Per-frame uniform buffer: mat4 (64) + vec4 (16) = 80 bytes.
    this.frameUBO = this.device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.frameBG = this.device.createBindGroup({
      layout: this.frameBGL,
      entries: [{ binding: 0, resource: { buffer: this.frameUBO } }],
    });
  }

  uploadMeshes(meshes) {
    const dev = this.device;
    this.objects = meshes.map((m) => {
      const vbo = dev.createBuffer({
        size: m.interleaved.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(vbo, 0, m.interleaved);

      const triBuf = dev.createBuffer({
        size: align4(m.triIndices.byteLength),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(triBuf, 0, m.triIndices);

      const lineBuf = dev.createBuffer({
        size: align4(m.lineIndices.byteLength),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(lineBuf, 0, m.lineIndices);

      // Per-object UBO: model (64) + color (16).
      const ubo = dev.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(ubo, 64, new Float32Array([...m.color, 1.0]));
      const bg = dev.createBindGroup({
        layout: this.objBGL,
        entries: [{ binding: 0, resource: { buffer: ubo } }],
      });

      return {
        vbo,
        triBuf,
        lineBuf,
        ubo,
        bg,
        triCount: m.triIndices.length,
        lineCount: m.lineIndices.length,
      };
    });
  }

  // Set up the dynamic soft-body debug mesh (beams drawn as lines). Vertices are
  // updated every frame from the node positions; topology (line indices) is fixed.
  setSoftBody({ nodeCount, lineIndices, color }) {
    const dev = this.device;
    const vbo = dev.createBuffer({
      size: nodeCount * 24, // 6 floats (pos+normal) per node
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const lineBuf = dev.createBuffer({
      size: align4(lineIndices.byteLength),
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(lineBuf, 0, lineIndices);

    const ubo = dev.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Identity model + color (model in first 64 bytes, color at offset 64).
    const ident = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    dev.queue.writeBuffer(ubo, 0, ident);
    dev.queue.writeBuffer(ubo, 64, new Float32Array([...color, 1.0]));
    const bg = dev.createBindGroup({
      layout: this.objBGL,
      entries: [{ binding: 0, resource: { buffer: ubo } }],
    });

    this.soft = { vbo, lineBuf, bg, lineCount: lineIndices.length };
  }

  // Dynamic skinned car body: static triangle topology, per-frame vertices.
  setBody({ maxVerts, triIndices, color }) {
    const dev = this.device;
    const vbo = dev.createBuffer({
      size: maxVerts * 24,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const triBuf = dev.createBuffer({
      size: align4(triIndices.byteLength),
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(triBuf, 0, triIndices);
    const ubo = dev.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const ident = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    dev.queue.writeBuffer(ubo, 0, ident);
    dev.queue.writeBuffer(ubo, 64, new Float32Array([...color, 1.0]));
    const bg = dev.createBindGroup({
      layout: this.objBGL,
      entries: [{ binding: 0, resource: { buffer: ubo } }],
    });
    this.body = { vbo, triBuf, bg, triCount: triIndices.length };
  }

  setSize(w, h) {
    // Canvas backing store is sized by the caller; (re)create the depth target.
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  render(viewProj, models, opts) {
    const dev = this.device;
    // Frame uniforms.
    const frameData = new Float32Array(20);
    frameData.set(viewProj, 0);
    frameData.set([LIGHT_DIR[0], LIGHT_DIR[1], LIGHT_DIR[2], 0.0], 16);
    dev.queue.writeBuffer(this.frameUBO, 0, frameData);

    // Per-object model matrices.
    for (let i = 0; i < this.objects.length; i++) {
      dev.queue.writeBuffer(this.objects[i].ubo, 0, models[i]);
    }

    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.53, g: 0.68, b: 0.86, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    const wire = opts && opts.wireframe;
    pass.setPipeline(wire ? this.wirePipeline : this.solidPipeline);
    pass.setBindGroup(0, this.frameBG);
    for (const o of this.objects) {
      pass.setBindGroup(1, o.bg);
      pass.setVertexBuffer(0, o.vbo);
      if (wire) {
        pass.setIndexBuffer(o.lineBuf, "uint16");
        pass.drawIndexed(o.lineCount);
      } else {
        pass.setIndexBuffer(o.triBuf, "uint16");
        pass.drawIndexed(o.triCount);
      }
    }

    // Skinned car body (solid, no cull). Vertices follow the chassis nodes.
    if (this.body && opts && opts.body) {
      dev.queue.writeBuffer(this.body.vbo, 0, opts.body.interleaved);
      pass.setPipeline(this.bodyPipeline);
      pass.setBindGroup(0, this.frameBG);
      pass.setBindGroup(1, this.body.bg);
      pass.setVertexBuffer(0, this.body.vbo);
      pass.setIndexBuffer(this.body.triBuf, "uint16");
      pass.drawIndexed(this.body.triCount);
    }

    // Soft-body debug lines (always drawn as lines, regardless of wireframe).
    // Vertices follow the nodes; the index buffer holds only unbroken beams.
    if (this.soft && opts && opts.soft) {
      dev.queue.writeBuffer(this.soft.vbo, 0, opts.soft.interleaved);
      const count = opts.soft.lineCount;
      if (count > 0) {
        dev.queue.writeBuffer(this.soft.lineBuf, 0, opts.soft.lineIndices);
        pass.setPipeline(this.wirePipeline);
        pass.setBindGroup(0, this.frameBG);
        pass.setBindGroup(1, this.soft.bg);
        pass.setVertexBuffer(0, this.soft.vbo);
        pass.setIndexBuffer(this.soft.lineBuf, "uint16");
        pass.drawIndexed(count);
      }
    }

    pass.end();
    dev.queue.submit([encoder.finish()]);
  }
}

function align4(n) {
  return (n + 3) & ~3;
}
