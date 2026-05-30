// Game bootstrap + render loop. Owns the WASM world, the active render backend,
// input, and the per-frame data hand-off (zero-copy Float32Array over WASM
// linear memory).

import init, { World } from "./pkg/web_3d_car.js";
import { createRenderer } from "./renderer.js";
import { buildMesh } from "./meshes.js";
import { buildCarBody } from "./carbody.js";
import { Input } from "./input.js";
import { UI } from "./ui.js";
import { perspective, multiply, deg2rad } from "./math.js";

async function main() {
  const wasm = await init(); // InitOutput; exposes `.memory`
  const world = new World();

  // The descriptor (geometry + colors) is the single source of truth for sizes.
  const descriptor = JSON.parse(world.descriptor);
  const carDesc = JSON.parse(world.car_descriptor);
  const meshes = descriptor.map(buildMesh);
  // Append one cylinder mesh per wheel; they render with WASM model matrices
  // (positioned at the hub nodes, steered + spun).
  for (let i = 0; i < carDesc.wheelCount; i++) {
    meshes.push(
      buildMesh({
        kind: "cylinder",
        radius: carDesc.wheelRadius,
        halfWidth: carDesc.wheelHalfWidth,
        color: carDesc.wheelColor,
      })
    );
  }
  const totalTris = meshes.reduce((a, m) => a + m.triCount, 0);
  const objCount = world.object_count(); // static + wheels = meshes.length

  const canvas = document.getElementById("gl");
  const ui = new UI();
  const { backend, name, zeroToOne } = await createRenderer(canvas, ui.forcedBackend());
  ui.setBackendName(name);
  backend.uploadMeshes(meshes);

  // Soft-body debug mesh: beams drawn as lines, vertices follow the nodes.
  const soft = JSON.parse(world.soft_descriptor);
  const softLineIndices = new Uint16Array(soft.beams);
  backend.setSoftBody({
    nodeCount: soft.nodeCount,
    lineIndices: softLineIndices,
    color: soft.color,
  });
  // Skinned car body (FFD over the 8 chassis nodes).
  const body = buildCarBody(carDesc);
  backend.setBody({
    maxVerts: body.vCount,
    triIndices: body.triIndices,
    color: body.color,
  });

  // Scratch interleaved [px,py,pz, nx,ny,nz] per node; normals preset to up so
  // the debug lines render at near-full brightness.
  const softScratch = new Float32Array(soft.nodeCount * 6);
  for (let i = 0; i < soft.nodeCount; i++) {
    softScratch[i * 6 + 4] = 1.0; // ny = 1
  }
  let nodeView = null;
  let lineView = null;
  const refreshNodeView = () => {
    nodeView = new Float32Array(
      wasm.memory.buffer,
      world.node_buffer_ptr(),
      world.node_buffer_len()
    );
    lineView = new Uint16Array(
      wasm.memory.buffer,
      world.line_buffer_ptr(),
      world.line_buffer_len()
    );
  };

  const input = new Input(canvas);

  // Zero-copy view over the shared buffer. Re-created if WASM memory grows
  // (which detaches the old ArrayBuffer).
  let bufView = null;
  const refreshView = () => {
    bufView = new Float32Array(
      wasm.memory.buffer,
      world.buffer_ptr(),
      world.buffer_len()
    );
  };

  // Size the canvas backing store = CSS size * render scale.
  const resize = () => {
    const scale = ui.state.renderScale;
    const w = Math.max(1, Math.round(canvas.clientWidth * scale));
    const h = Math.max(1, Math.round(canvas.clientHeight * scale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      backend.setSize(w, h);
    }
  };
  resize();
  window.addEventListener("resize", resize);
  ui.onChange(resize);

  refreshView();
  refreshNodeView();
  let last = performance.now();
  let fpsTimer = 0,
    fpsFrames = 0,
    fps = 0,
    lastMs = 0;

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    lastMs = dt * 1000;
    resize();

    // Drive the simulation.
    const d = input.driver();
    world.set_input(d.throttle, d.brake, d.steer, d.handbrake, d.reset);
    const orbit = input.takeOrbit();
    world.step(dt, input.cameraMode, orbit.dx, orbit.dy);

    // Read view + model matrices (zero-copy). Memory growth detaches buffers.
    if (!bufView || bufView.buffer !== wasm.memory.buffer || bufView.length === 0) {
      refreshView();
      refreshNodeView();
    }
    const view = bufView.subarray(0, 16);
    const models = [];
    for (let i = 0; i < objCount; i++) {
      models.push(bufView.subarray(16 * (i + 1), 16 * (i + 2)));
    }

    // Pack current node positions into the soft-body scratch buffer.
    for (let i = 0; i < soft.nodeCount; i++) {
      softScratch[i * 6] = nodeView[i * 3];
      softScratch[i * 6 + 1] = nodeView[i * 3 + 1];
      softScratch[i * 6 + 2] = nodeView[i * 3 + 2];
    }

    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = perspective(deg2rad(ui.state.fovDeg), aspect, 0.1, 500, zeroToOne);
    const viewProj = multiply(proj, view);

    // Skin the body to the chassis nodes (indices 0..chassisCount).
    const bodyInterleaved = body.skin(nodeView.subarray(0, carDesc.chassisCount * 3));

    const renderOpts = {
      wireframe: ui.state.wireframe,
      body: { interleaved: bodyInterleaved },
    };
    // Node/beam structure overlay (toggle) — useful for seeing deformation.
    if (ui.state.showStructure) {
      const lineCount = world.line_count();
      renderOpts.soft = {
        interleaved: softScratch,
        lineIndices: lineView.subarray(0, lineCount),
        lineCount,
      };
    }
    backend.render(viewProj, models, renderOpts);

    // HUD (throttled).
    fpsTimer += dt;
    fpsFrames++;
    if (fpsTimer >= 0.25) {
      fps = fpsFrames / fpsTimer;
      fpsTimer = 0;
      fpsFrames = 0;
      ui.updateHUD({
        fps,
        ms: lastMs,
        tris: totalTris,
        speed: world.speed_kmh(),
        rpm: world.rpm(),
        gear: world.gear(),
        cam: input.cameraMode,
        resW: canvas.width,
        resH: canvas.height,
      });
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  const ov = document.getElementById("overlay");
  if (ov) {
    ov.style.display = "flex";
    ov.textContent = "Failed to start: " + e.message;
  }
});
