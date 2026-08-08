/* AudioWorkletProcessor: mic input -> 16-bit PCM frames posted to the main thread.
   Runs on the audio rendering thread; must stay allocation-light per callback.

   Noise gate: Deepgram's speech_final/UtteranceEnd endpointing looks at
   whether the audio it *receives* goes quiet, not whether it's decoded
   real words — continuous background noise (fan, room hum, traffic) can
   keep the input "loud enough" that Deepgram never sees silence, so it
   never realizes you've stopped talking. This tracks a slowly-adapting
   noise-floor estimate and zeroes out blocks that are only ambient noise
   before sending them, so Deepgram (and everything downstream — including
   our own semantic turn-check, which only runs once speech_final fires)
   sees clean silence during real pauses regardless of room noise. Still
   sends every block (never stops streaming) — only the *content* changes,
   which matters since Deepgram's timing is based on continuous audio, not
   wall-clock time (see TRAILING_SILENCE_MS in test_voice_latency.py). */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.noiseFloor = 0.005;
    this.gateOpen = false;
    this.holdBlocks = 0;
    // Pre-roll ring buffer: when the gate opens we flush these *previous*
    // blocks first, so the attack of the first word isn't clipped. Classic
    // failure mode of naive energy gates — the gate opens a few ms into
    // "Hello", so STT receives "ello" (or nothing recognizable).
    this.preRoll = [];
    this.frames = 0;
    this.gatedFrames = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || !channel.length) return true;

    let sumSquares = 0;
    for (let i = 0; i < channel.length; i++) sumSquares += channel[i] * channel[i];
    const rms = Math.sqrt(sumSquares / channel.length);

    // Adapt toward quiet blocks only. Never let real speech drag the floor
    // up, and HARD-CAP the floor low: an over-high floor makes the gate
    // latch shut and swallow speech entirely (the server then never sees
    // end-of-speech and never replies), which is far worse than leaking
    // some noise through — a leaky gate just means endpointing fires late.
    // This gate must fail OPEN, never closed.
    if (rms < this.noiseFloor * 1.5) {
      this.noiseFloor += (rms - this.noiseFloor) * 0.05;
    }
    this.noiseFloor = Math.max(0.001, Math.min(0.02, this.noiseFloor));

    // Absolute floor on the threshold too, so even a pathological noiseFloor
    // estimate can't push the open-threshold above normal speech level.
    const openThreshold = Math.min(0.05, Math.max(0.004, this.noiseFloor * 2.5));
    const closeThreshold = openThreshold * 0.6;

    if (rms > openThreshold) {
      this.gateOpen = true;
      this.holdBlocks = 25; // ~200ms hold, so brief inter-word dips stay open
    } else if (this.gateOpen && rms < closeThreshold) {
      if (this.holdBlocks > 0) this.holdBlocks--;
      else this.gateOpen = false;
    }

    const toInt16 = (src) => {
      const out = new Int16Array(src.length);
      for (let i = 0; i < src.length; i++) {
        const s = Math.max(-1, Math.min(1, src[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return out;
    };

    this.frames++;
    if (this.gateOpen) {
      // Flush pre-roll first so the word's onset survives.
      for (const buffered of this.preRoll) {
        this.port.postMessage(buffered.buffer, [buffered.buffer]);
      }
      this.preRoll.length = 0;
      const pcm16 = toInt16(channel);
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    } else {
      this.gatedFrames++;
      // Keep ~100ms of recent audio as pre-roll for the next gate opening.
      this.preRoll.push(toInt16(channel));
      if (this.preRoll.length > 12) this.preRoll.shift();
      // Still stream silence: Deepgram's endpointing advances on elapsed
      // AUDIO time, not wall-clock, so the stream must never stop.
      this.port.postMessage(new Int16Array(channel.length).buffer);
    }

    // Periodic telemetry so a stuck-shut gate is diagnosable instead of
    // presenting as a mysteriously unresponsive agent.
    if (this.frames % 250 === 0) {
      this.port.postMessage({
        type: "gate_stats",
        gatedRatio: this.gatedFrames / 250,
        noiseFloor: this.noiseFloor,
        openThreshold,
        lastRms: rms,
      });
      this.gatedFrames = 0;
    }
    return true; // keep the processor alive for the life of the call
  }
}

registerProcessor("capture-processor", CaptureProcessor);
