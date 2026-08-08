/* AudioWorkletProcessor: queued PCM16 playback with instant-flush for barge-in.
   Main thread pushes chunks via port.postMessage({type:'push', buffer}); a
   {type:'clear'} message (sent the moment the server reports an interruption)
   drops everything queued so playback stops within one render quantum.

   Chunks arrive over two network hops (Deepgram -> our server -> browser) in
   bursts, not a steady drip — measured real gaps of ~200ms between clusters
   of chunks, not just the ~20-30ms steady-state pacing. Playing whatever's
   queued the instant any audio shows up means those gaps produce audible
   glitches. A lead-in buffer (~300ms, comfortably above the observed gap
   size) absorbs that jitter into a brief, silent pause instead — re-armed
   any time the queue genuinely runs dry, so a gap causes a clean pause
   rather than a stutter. */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.readOffset = 0;
    this.queuedSamples = 0;
    this._hasAudio = false;
    this.buffering = true;
    this.minBufferSamples = Math.round(sampleRate * 0.3);
    // Paused != cleared. On the first hint the user may be talking (mic
    // energy) we pause instantly so we stop talking over them, but KEEP the
    // queued audio — if it turns out to be noise rather than speech we
    // resume seamlessly instead of having destroyed the reply.
    this.paused = false;

    this.port.onmessage = (event) => {
      const { type, buffer } = event.data;
      if (type === "push") {
        const chunk = new Int16Array(buffer);
        this.queue.push(chunk);
        this.queuedSamples += chunk.length;
      } else if (type === "clear") {
        this.queue = [];
        this.readOffset = 0;
        this.queuedSamples = 0;
        this.buffering = true;
        this.paused = false;
      } else if (type === "pause") {
        this.paused = true;
      } else if (type === "resume") {
        this.paused = false;
      }
    };
  }

  _setHasAudio(hasAudio) {
    if (hasAudio !== this._hasAudio) {
      this._hasAudio = hasAudio;
      this.port.postMessage({ type: hasAudio ? "playback_active" : "playback_idle" });
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];

    if (this.paused) {
      output.fill(0);
      return true; // hold position in the queue; resume picks up where we left off
    }

    if (this.buffering && this.queuedSamples < this.minBufferSamples) {
      output.fill(0);
      this._setHasAudio(false);
      return true;
    }
    this.buffering = false;

    for (let i = 0; i < output.length; i++) {
      if (this.queue.length === 0) {
        this.buffering = true; // ran dry — re-buffer instead of stuttering
        output.fill(0, i);
        break;
      }
      const current = this.queue[0];
      if (this.readOffset >= current.length) {
        this.queue.shift();
        this.readOffset = 0;
        i--; // retry this sample against the next chunk
        continue;
      }
      output[i] = current[this.readOffset++] / 32768;
      this.queuedSamples--;
    }

    this._setHasAudio(this.queue.length > 0);
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
