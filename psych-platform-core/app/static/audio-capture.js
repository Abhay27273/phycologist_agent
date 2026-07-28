/* AudioWorkletProcessor: mic input -> 16-bit PCM frames posted to the main thread.
   Runs on the audio rendering thread; must stay allocation-light per callback. */
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      const pcm16 = new Int16Array(channel.length);
      for (let i = 0; i < channel.length; i++) {
        const s = Math.max(-1, Math.min(1, channel[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]); // transfer, zero-copy
    }
    return true; // keep the processor alive for the life of the call
  }
}

registerProcessor("capture-processor", CaptureProcessor);
