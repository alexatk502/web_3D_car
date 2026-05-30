# WASM 3D Car

A browser 3D driving sandbox. **All physics runs in Rust/WASM** (Rapier's raycast
vehicle); **rendering is triangle-based on the GPU** (WebGPU, with a WebGL2
fallback). Plain HTML/JS/CSS shell.

## What it does

- 3D chase / hood / orbit cameras with a controllable **FOV** and perspective projection.
- A car with **acceleration, braking/reverse, steering, and a handbrake**.
- **Collision** with the ground (raycast suspension) and with scattered obstacle boxes.
- Runtime settings: **render resolution** (scale), **FOV**, **wireframe** toggle, and a
  live HUD (FPS, frame ms, triangle count, speed).

## Architecture

| Layer | Owns |
| --- | --- |
| **Rust → WASM** (`src/`) | Rapier physics world, raycast vehicle, camera math. Writes a flat `f32` buffer: `[view][model_0…model_n]`. |
| **JS** (`www/`) | WebGPU/WebGL2 rendering, input, settings UI, the rAF loop. Reads the WASM buffer **zero-copy** via a `Float32Array` over linear memory; builds the (backend-specific) projection and `viewProj = proj · view`. |
| **HTML/CSS** | Canvas + settings panel + HUD. |

The scene descriptor returned by WASM (`world.descriptor`) is the single source of
truth for object sizes/colors, so JS meshes always match the Rapier colliders.

Key files: `src/{lib,physics,vehicle,camera,scene,render_state}.rs`,
`www/{main,renderer,webgpu_backend,webgl_backend,meshes,input,ui,math}.js`,
`www/{shader.wgsl,shaders.glsl.js}`.

## Build & run

Prerequisites: Rust + `wasm32-unknown-unknown`, `wasm-pack`, Python 3 (to serve).

```bash
./build.sh          # wasm-pack build --target web --out-dir www/pkg
./serve.sh          # python3 -m http.server 8080 --directory www
# open http://localhost:8080  (a WebGPU-capable browser: recent Chrome/Edge/Firefox)
```

`file://` will not work — ES modules and the WASM fetch require http; `localhost`
is a secure context, so WebGPU is enabled there.

## Controls

| Key | Action |
| --- | --- |
| `W` / `↑` | accelerate |
| `S` / `↓` | brake / reverse |
| `A` `D` / `←` `→` | steer |
| `Space` | handbrake |
| `R` | reset car |
| `C` | cycle camera (chase → hood → orbit) |
| drag | rotate orbit camera (orbit mode) |

## Settings

- **Render scale** — internal resolution multiplier (0.25×–1×). Lower = pixelated + faster.
- **FOV** — vertical field of view (40°–110°), feeds the projection matrix.
- **Wireframe** — draw triangle edges instead of filled faces.
- **Renderer** — Auto (WebGPU→WebGL2) or force one via `?backend=webgl|webgpu`.

## Tuning the car feel

Vehicle constants live at the top of `src/vehicle.rs` (`ENGINE_FORCE`, `BRAKE_FORCE`,
`MAX_STEER`, suspension stiffness/damping, chassis density). Adjust and re-run
`./build.sh`. If steering feels inverted, flip the sign of `steer` in `Input::driver`
(`www/input.js`) or in the wheel steering assignment.
