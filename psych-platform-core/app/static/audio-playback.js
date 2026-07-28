/* AudioWorkletProcessor: queued PCM16 playback with instant-flush for barge-in.
   Main thread pushes chunks via port.postMessage({type:'push', buffer}); a
   {type:'clear'} message (sent the moment the server reports an interruption)
   drops everything queued so playback stops within one render quantum. */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.readOffset = 0;
    this._hasAudio = false;

    this.port.onmessage = (event) => {
      const { type, buffer } = event.data;
      if (type === "push") {
        this.queue.push(new Int16Array(buffer));
      } else if (type === "clear") {
        this.queue = [];
        this.readOffset = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    for (let i = 0; i < output.length; i++) {
      if (this.queue.length === 0) {
        output[i] = 0;
        continue;
      }
      const current = this.queue[0];
      if (this.readOffset >= current.length) {
        this.queue.shift();
        this.readOffset = 0;
        i--; // retry this sample against the next chunk
        continue;
      }
      output[i] = current[this.readOffset++] / 32768;
    }

    const hasAudio = this.queue.length > 0;
    if (hasAudio !== this._hasAudio) {
      this._hasAudio = hasAudio;
      this.port.postMessage({ type: hasAudio ? "playback_active" : "playback_idle" });
    }
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
