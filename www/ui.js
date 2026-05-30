// Settings panel + HUD wiring. Holds the live settings state the game loop reads
// each frame and renders the on-screen telemetry.

const CAM_NAMES = ["Chase", "Hood", "Orbit"];

export class UI {
  constructor() {
    this.state = {
      renderScale: 1.0,
      fovDeg: 70,
      wireframe: false,
      showStructure: false,
    };
    this._changeCbs = [];

    this.el = {
      scale: document.getElementById("scale"),
      scaleVal: document.getElementById("scaleVal"),
      fov: document.getElementById("fov"),
      fovVal: document.getElementById("fovVal"),
      wire: document.getElementById("wire"),
      struct: document.getElementById("struct"),
      backendSel: document.getElementById("backendSel"),
      backendName: document.getElementById("backendName"),
      hudFps: document.getElementById("hudFps"),
      hudMs: document.getElementById("hudMs"),
      hudTris: document.getElementById("hudTris"),
      hudSpeed: document.getElementById("hudSpeed"),
      hudRpm: document.getElementById("hudRpm"),
      hudGear: document.getElementById("hudGear"),
      hudCam: document.getElementById("hudCam"),
      hudRes: document.getElementById("hudRes"),
    };

    // Initialize controls from defaults.
    this.el.scale.value = String(this.state.renderScale);
    this.el.fov.value = String(this.state.fovDeg);
    this.el.wire.checked = this.state.wireframe;
    this.el.struct.checked = this.state.showStructure;
    this.el.backendSel.value = this.forcedBackend() || "auto";

    this.el.scale.addEventListener("input", () => {
      this.state.renderScale = parseFloat(this.el.scale.value);
      this.el.scaleVal.textContent = this.state.renderScale.toFixed(2) + "x";
      this._emitChange();
    });
    this.el.fov.addEventListener("input", () => {
      this.state.fovDeg = parseFloat(this.el.fov.value);
      this.el.fovVal.textContent = this.state.fovDeg + "°";
    });
    this.el.wire.addEventListener("change", () => {
      this.state.wireframe = this.el.wire.checked;
    });
    this.el.struct.addEventListener("change", () => {
      this.state.showStructure = this.el.struct.checked;
    });
    // Changing the backend reloads with a ?backend= override.
    this.el.backendSel.addEventListener("change", () => {
      const v = this.el.backendSel.value;
      window.location.search = v === "auto" ? "" : "?backend=" + v;
    });

    this.el.scaleVal.textContent = this.state.renderScale.toFixed(2) + "x";
    this.el.fovVal.textContent = this.state.fovDeg + "°";
  }

  // Reads ?backend=webgl|webgpu from the URL (used to force the fallback path).
  forcedBackend() {
    const v = new URLSearchParams(window.location.search).get("backend");
    return v === "webgl" || v === "webgpu" ? v : null;
  }

  onChange(cb) {
    this._changeCbs.push(cb);
  }
  _emitChange() {
    for (const cb of this._changeCbs) cb();
  }

  setBackendName(name) {
    this.el.backendName.textContent = name;
  }

  updateHUD({ fps, ms, tris, speed, rpm, gear, cam, resW, resH }) {
    this.el.hudFps.textContent = fps.toFixed(0);
    this.el.hudMs.textContent = ms.toFixed(1);
    this.el.hudTris.textContent = tris.toLocaleString();
    this.el.hudSpeed.textContent = speed.toFixed(0);
    this.el.hudRpm.textContent = rpm.toFixed(0);
    this.el.hudGear.textContent = gear;
    this.el.hudCam.textContent = CAM_NAMES[cam] || "?";
    this.el.hudRes.textContent = resW + "×" + resH;
  }
}
