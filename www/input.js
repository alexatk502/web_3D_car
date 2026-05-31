// Keyboard + pointer input. Produces the normalized driver input the WASM world
// expects, tracks the camera mode (C cycles), and accumulates orbit-drag deltas.

export class Input {
  constructor(canvas) {
    this.keys = new Set();
    this.cameraMode = 0; // 0 chase, 1 hood, 2 orbit
    this.orbitDx = 0;
    this.orbitDy = 0;
    this._dragging = false;
    this._lastX = 0;
    this._lastY = 0;
    // Edge-triggered gearbox actions, consumed once per frame by takeShift().
    this._gearUp = false;
    this._gearDown = false;
    this._toggleManual = false;

    const block = new Set([
      "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space",
    ]);
    window.addEventListener("keydown", (e) => {
      if (block.has(e.code)) e.preventDefault();
      if (e.code === "KeyC" && !e.repeat) {
        this.cameraMode = (this.cameraMode + 1) % 3;
      }
      // Gearbox (edge-triggered): E upshift, Q downshift, M toggle auto/manual.
      if (e.code === "KeyE" && !e.repeat) this._gearUp = true;
      if (e.code === "KeyQ" && !e.repeat) this._gearDown = true;
      if (e.code === "KeyM" && !e.repeat) this._toggleManual = true;
      this.keys.add(e.code);
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));

    // Pointer drag rotates the orbit camera (only meaningful in orbit mode).
    canvas.addEventListener("pointerdown", (e) => {
      this._dragging = true;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this._dragging) return;
      this.orbitDx += (e.clientX - this._lastX) * 0.01;
      this.orbitDy += (e.clientY - this._lastY) * 0.01;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
    });
    const stop = (e) => {
      this._dragging = false;
      if (canvas.hasPointerCapture && e.pointerId !== undefined &&
          canvas.hasPointerCapture(e.pointerId)) {
        canvas.releasePointerCapture(e.pointerId);
      }
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
  }

  has(...codes) {
    return codes.some((c) => this.keys.has(c));
  }

  // Returns the current driver input. `throttle`/`brake` are 0..1, `steer` is
  // -1..1 (positive = left).
  driver() {
    const throttle = this.has("KeyW", "ArrowUp") ? 1 : 0;
    const brake = this.has("KeyS", "ArrowDown") ? 1 : 0;
    let steer = 0;
    if (this.has("KeyA", "ArrowLeft")) steer -= 1;
    if (this.has("KeyD", "ArrowRight")) steer += 1;
    const handbrake = this.has("Space");
    const reset = this.has("KeyR");
    // Clutch pedal: hold Shift to disengage (manual mode).
    const clutch = this.has("ShiftLeft", "ShiftRight") ? 1 : 0;
    return { throttle, brake, steer, handbrake, reset, clutch };
  }

  // Consume the edge-triggered gearbox actions (called once per frame).
  takeShift() {
    const s = { up: this._gearUp, down: this._gearDown, toggleManual: this._toggleManual };
    this._gearUp = false;
    this._gearDown = false;
    this._toggleManual = false;
    return s;
  }

  // Consume the accumulated orbit deltas (called once per frame).
  takeOrbit() {
    const d = { dx: this.orbitDx, dy: this.orbitDy };
    this.orbitDx = 0;
    this.orbitDy = 0;
    return d;
  }
}
