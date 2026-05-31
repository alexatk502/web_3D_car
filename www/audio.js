// Web Audio: a synthesized engine note (pitch tracks RPM) + impact/crash bursts.
// Created lazily — `ensure()` must first run inside (or after) a user gesture
// because of browser autoplay policy. `update(rpm, impact, dt)` is called each
// frame; `silence()` fades it out when sound is toggled off.

export class Audio {
  constructor() {
    this.ctx = null;
    this.ready = false;
    this._impactCooldown = 0;
  }

  // Build the graph (idempotent) and resume the context. Call from a gesture.
  ensure() {
    if (this.ctx) {
      if (this.ctx.state === "suspended") this.ctx.resume();
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    this.ctx = ctx;

    this.master = ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(ctx.destination);

    // Engine: two slightly-detuned sawtooths through a lowpass → gain.
    this.engGain = ctx.createGain();
    this.engGain.gain.value = 0.0;
    this.lp = ctx.createBiquadFilter();
    this.lp.type = "lowpass";
    this.lp.frequency.value = 800;
    this.lp.connect(this.engGain);
    this.engGain.connect(this.master);

    this.osc1 = ctx.createOscillator();
    this.osc1.type = "sawtooth";
    this.osc2 = ctx.createOscillator();
    this.osc2.type = "sawtooth";
    this.osc2.detune.value = 14; // beat between the two for a richer note
    this.osc1.connect(this.lp);
    this.osc2.connect(this.lp);
    this.osc1.start();
    this.osc2.start();

    // Pre-built white-noise buffer reused for every impact burst.
    const n = Math.floor(ctx.sampleRate * 0.4);
    const buf = ctx.createBuffer(1, n, ctx.sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = Math.random() * 2 - 1;
    this.noiseBuf = buf;

    this.ready = true;
  }

  update(rpm, impact, dt) {
    if (!this.ready) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    // Pitch from RPM: idle ~900 → ~68 Hz, redline ~6800 → ~334 Hz.
    const f = 28 + rpm * 0.045;
    this.osc1.frequency.setTargetAtTime(f, t, 0.02);
    this.osc2.frequency.setTargetAtTime(f, t, 0.02);
    // Brighten + lift volume with revs.
    this.lp.frequency.setTargetAtTime(500 + rpm * 0.35, t, 0.05);
    const target = 0.04 + Math.min(rpm / 6800, 1) * 0.1;
    this.engGain.gain.setTargetAtTime(target, t, 0.05);

    // Impact burst when the impact signal spikes (debounced).
    this._impactCooldown -= dt || 0.016;
    if (impact > 4 && this._impactCooldown <= 0) {
      this._playImpact(Math.min(impact / 20, 1));
      this._impactCooldown = 0.08;
    }
  }

  _playImpact(vol) {
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuf;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = 110 + vol * 220;
    bp.Q.value = 0.7;
    const g = ctx.createGain();
    g.gain.setValueAtTime(vol * 0.9, t);
    g.gain.exponentialRampToValueAtTime(0.0008, t + 0.18);
    src.connect(bp);
    bp.connect(g);
    g.connect(this.master);
    src.start(t);
    src.stop(t + 0.2);
  }

  // Fade the engine note out (sound toggled off).
  silence() {
    if (this.ready) {
      this.engGain.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.05);
    }
  }
}
