// Game bootstrap + render loop. Owns the WASM world, the active render backend,
// input, and the per-frame data hand-off (zero-copy Float32Array over WASM
// linear memory).

import init, { World, initThreadPool } from "./pkg/web_3d_car.js";
import { createRenderer } from "./renderer.js";
import { buildMesh } from "./meshes.js";
import { buildCarBody } from "./carbody.js";
import { buildTire } from "./tirebody.js";
import { Input } from "./input.js";
import { UI } from "./ui.js";
import { Audio } from "./audio.js";
import { perspective, multiply, deg2rad } from "./math.js";

async function main() {
  const wasm = await init(); // InitOutput; exposes `.memory`

  // Spin up the WASM thread pool (Web Workers + SharedArrayBuffer) for the
  // parallel solver. Requires cross-origin isolation (serve.py sets COOP/COEP);
  // if that's missing the pool can't start, so we fall back to single-threaded.
  let threads = 1;
  if (self.crossOriginIsolated) {
    threads = Math.max(1, navigator.hardwareConcurrency || 1);
    try {
      await initThreadPool(threads);
      console.log(`[threads] pool started with ${threads} workers`);
    } catch (e) {
      threads = 1;
      console.warn("[threads] pool init failed, running single-threaded:", e);
    }
  } else {
    console.warn(
      "[threads] not cross-origin isolated (use ./serve.py) — single-threaded"
    );
  }

  const world = new World();
  // Enable the parallel solver path only when the worker pool actually started.
  world.set_threaded(threads > 1);

  // The descriptor (geometry + colors) is the single source of truth for sizes.
  const descriptor = JSON.parse(world.descriptor);
  const carDesc = JSON.parse(world.car_descriptor);
  const staticMeshes = descriptor.map(buildMesh); // terrain + obstacles + patches

  const canvas = document.getElementById("gl");
  const ui = new UI();
  const { backend, name, zeroToOne } = await createRenderer(canvas, ui.forcedBackend());
  ui.setBackendName(name);

  const input = new Input(canvas);
  const audio = new Audio();

  // --- World-dependent state, rebuilt whenever the car count changes (spawn). ---
  let objCount = 0; // static + 4 wheels per car
  let totalTris = 0;
  let carCount = 0;
  let bodies = []; // one skinnable body instance per car (own vertex buffers)
  let tires = []; // one tire mesh per wheel (carCount*wheelsPerCar) when deformable
  let soft = null; // parsed combined soft descriptor
  let softScratch = null; // interleaved [px,py,pz, nx,ny,nz] for all nodes
  let nodeView = null;
  let lineView = null;
  let bufView = null;
  let lastDeformable = null; // detect the deformable-tire toggle changing
  const wheelsPerCar = carDesc.wheelCount;
  const chassisCount = carDesc.chassisCount;
  const treadN = carDesc.treadN;

  const refreshNodeView = () => {
    nodeView = new Float32Array(
      wasm.memory.buffer,
      world.node_buffer_ptr(),
      world.node_buffer_len()
    );
    // Line indices are global node indices across all cars → 32-bit.
    lineView = new Uint32Array(
      wasm.memory.buffer,
      world.line_buffer_ptr(),
      world.line_buffer_len()
    );
  };
  const refreshView = () => {
    bufView = new Float32Array(
      wasm.memory.buffer,
      world.buffer_ptr(),
      world.buffer_len()
    );
  };

  // (Re)build all per-car-count GPU resources. Called at startup and after a spawn.
  const rebuildWorld = () => {
    carCount = world.car_count();
    objCount = world.object_count();
    const deformable = ui.state.deformableTires;
    lastDeformable = deformable;
    const wheelTotal = world.wheel_count();

    // Static meshes. In rigid-tire mode also append a wheel cylinder per wheel
    // (drawn via WASM model matrices); in deformable mode the tires are dynamic
    // skinned meshes instead, so the cylinders are omitted.
    const meshes = staticMeshes.slice();
    if (!deformable) {
      for (let i = 0; i < wheelTotal; i++) {
        meshes.push(
          buildMesh({
            kind: "cylinder",
            radius: carDesc.wheelRadius,
            halfWidth: carDesc.wheelHalfWidth,
            color: carDesc.wheelColor,
          })
        );
      }
    }
    totalTris = meshes.reduce((a, m) => a + m.triCount, 0);
    backend.uploadMeshes(meshes);

    // Combined soft-body line mesh (all cars' nodes; global indices).
    soft = JSON.parse(world.soft_descriptor);
    backend.setSoftBody({
      nodeCount: soft.nodeCount,
      lineIndices: new Uint32Array(soft.beams),
      color: soft.color,
    });
    softScratch = new Float32Array(soft.nodeCount * 6);
    for (let i = 0; i < soft.nodeCount; i++) softScratch[i * 6 + 4] = 1.0; // ny=1

    // One body instance per car (own reused interleaved buffer); all share the
    // same skinning data since the structure is identical.
    bodies = Array.from({ length: carCount }, () => buildCarBody(carDesc));
    backend.setBody({
      maxVerts: bodies[0].vCount,
      triIndices: bodies[0].triIndices,
      color: bodies[0].color,
      count: carCount,
    });

    // Deformable tire meshes: one per wheel, skinned from the tread ring nodes.
    if (deformable) {
      tires = Array.from({ length: wheelTotal }, () =>
        buildTire(treadN, carDesc.wheelHalfWidth, carDesc.wheelColor)
      );
      backend.setTire({
        maxVerts: tires[0].vCount,
        triIndices: tires[0].triIndices,
        color: carDesc.wheelColor,
        count: wheelTotal,
      });
    } else {
      tires = [];
    }

    refreshView();
    refreshNodeView();
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

  rebuildWorld();
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

    // Fleet actions: Tab switches the active car, B spawns one (then rebuild).
    const wld = input.takeWorld();
    if (wld.switchCar && world.car_count() > 1) {
      world.set_active((world.active_car() + 1) % world.car_count());
    }
    if (wld.spawnCar) {
      world.spawn_near_active(8.0, 3.0);
      rebuildWorld();
    }
    // Rebuild if the deformable-tire toggle changed.
    if (ui.state.deformableTires !== lastDeformable) {
      rebuildWorld();
    }

    // Drive the simulation.
    const d = input.driver();
    const sh = input.takeShift();
    if (sh.toggleManual) world.toggle_manual();
    if (sh.up) world.shift_up();
    if (sh.down) world.shift_down();
    world.set_input(d.throttle, d.brake, d.steer, d.handbrake, d.reset, d.clutch);
    const orbit = input.takeOrbit();
    world.step(dt, input.cameraMode, orbit.dx, orbit.dy);

    // Audio (opt-in): engine note tracks rpm, bursts on impacts.
    if (ui.state.sound) {
      audio.ensure();
      audio.update(world.rpm(), world.impact_level(), dt);
    } else {
      audio.silence();
    }

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

    // Skin each car's body to its own chassis-node slice. Each `bodies[c]` is a
    // separate instance, so the returned interleaved buffers don't alias.
    const interleavedList = [];
    for (let c = 0; c < carCount; c++) {
      const off = world.car_node_offset(c);
      interleavedList.push(
        bodies[c].skin(nodeView.subarray(off * 3, (off + chassisCount) * 3))
      );
    }

    const renderOpts = {
      wireframe: ui.state.wireframe,
      body: { interleavedList },
    };

    // Deformable tires: skin each wheel's tread ring from its node slice. Per-car
    // node layout is [chassis…][hub, tread×treadN] per wheel, so wheel w of car c
    // starts at: car_offset + chassisCount + w*(1+treadN).
    if (tires.length > 0) {
      const tireList = [];
      for (let c = 0; c < carCount; c++) {
        const off = world.car_node_offset(c);
        for (let w = 0; w < wheelsPerCar; w++) {
          const hubG = off + chassisCount + w * (1 + treadN);
          const hub = [nodeView[hubG * 3], nodeView[hubG * 3 + 1], nodeView[hubG * 3 + 2]];
          const treadFlat = nodeView.subarray((hubG + 1) * 3, (hubG + 1 + treadN) * 3);
          tireList.push(tires[c * wheelsPerCar + w].skin(hub, treadFlat));
        }
      }
      renderOpts.tire = { interleavedList: tireList };
    }

    // Node/beam structure overlay (toggle) — useful for seeing deformation.
    if (ui.state.showStructure) {
      const lineCount = world.line_count();
      renderOpts.soft = {
        interleaved: softScratch,
        lineIndices: lineView.subarray(0, lineCount),
        lineCount,
        bands: world.line_band_counts(), // [green, yellow, orange, red] index counts
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
        threads,
        nodes: world.node_count(),
        beams: world.beam_count(),
        substeps: world.substeps_last_frame(),
        manual: world.is_manual(),
        clutch: world.clutch(),
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
