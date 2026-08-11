/* ============================================================
   Mindful — real-time voice call orchestration.
   Two AudioContexts (capture @16kHz for STT, playback @24kHz for TTS),
   each backed by an AudioWorklet (see audio-capture.js / audio-playback.js).
   Reads auth state from localStorage — the same keys app.js writes —
   rather than sharing app.js's closure, to keep the two files decoupled.
   ============================================================ */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const chatScreen = $("chat-screen");
  const voiceScreen = $("voice-screen");
  const callBtn = $("call-btn");
  const endCallBtn = $("voice-end-btn");
  const muteBtn = $("voice-mute-btn");
  const orb = $("voice-orb");
  const stateLabel = $("voice-state-label");
  const transcriptEl = $("voice-transcript");
  const riskBadge = $("voice-risk-badge");

  const call = {
    ws: null,
    captureCtx: null,
    playbackCtx: null,
    captureNode: null,
    playbackNode: null,
    analyser: null,
    micStream: null,
    levelRAF: null,
    muted: false,
    active: false,
  };

  function sessionKeyFor(userId) {
    return `mindful_session_${userId}`;
  }

  function authState() {
    const token = localStorage.getItem("mindful_token");
    const userId = localStorage.getItem("mindful_user_id");
    if (!token || !userId) return null;
    const sessionId = localStorage.getItem(sessionKeyFor(userId));
    if (!sessionId) return null;
    return { token, userId, sessionId };
  }

  function setOrbState(state) {
    orb.dataset.state = state;
  }

  function setLabel(text) {
    stateLabel.textContent = text;
  }

  function setTranscript(text) {
    transcriptEl.textContent = text || "";
  }

  function setRisk(level) {
    const labels = { LOW: "● steady", MEDIUM: "● elevated", HIGH: "● please reach out" };
    riskBadge.dataset.level = level;
    riskBadge.textContent = labels[level] || labels.LOW;
  }

  // ---------------------------------------------------------
  // Start / stop
  // ---------------------------------------------------------

  callBtn.addEventListener("click", startCall);
  endCallBtn.addEventListener("click", endCall);
  muteBtn.addEventListener("click", toggleMute);

  async function startCall() {
    const auth = authState();
    if (!auth) return;

    chatScreen.hidden = true;
    voiceScreen.hidden = false;
    setOrbState("idle");
    setLabel("Connecting…");
    setTranscript("");
    setRisk("LOW");
    call.active = true;

    try {
      call.micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,   // keep: TTS through speakers must not self-trigger
          noiseSuppression: true,   // keep: handles steady room noise upstream of our gate
          // autoGainControl OFF deliberately: AGC ramps gain up during
          // silence, which inflates the apparent noise floor and destabilises
          // energy-based gating/VAD downstream (the level the gate sees for
          // "quiet" keeps moving). Constant gain gives turn detection a
          // stable reference.
          autoGainControl: false,
        },
      });
    } catch {
      setLabel("Microphone access is required for a voice call.");
      setTimeout(endCall, 2500);
      return;
    }

    try {
      call.captureCtx = new AudioContext({ sampleRate: 16000 });
      await call.captureCtx.audioWorklet.addModule("/audio-capture.js");
      call.playbackCtx = new AudioContext({ sampleRate: 24000 });
      await call.playbackCtx.audioWorklet.addModule("/audio-playback.js");
    } catch (err) {
      setLabel("This browser can't run real-time voice (AudioWorklet unsupported).");
      setTimeout(endCall, 3000);
      return;
    }

    const source = call.captureCtx.createMediaStreamSource(call.micStream);
    call.captureNode = new AudioWorkletNode(call.captureCtx, "capture-processor");
    source.connect(call.captureNode);

    call.analyser = call.captureCtx.createAnalyser();
    call.analyser.fftSize = 512;
    source.connect(call.analyser);
    startLevelMeter();

    call.playbackNode = new AudioWorkletNode(call.playbackCtx, "playback-processor", {
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    call.playbackNode.connect(call.playbackCtx.destination);
    call.playbackNode.port.onmessage = (event) => {
      if (event.data.type === "playback_active") setOrbState("speaking");
      else if (event.data.type === "playback_idle" && call.active) setOrbState("listening");
    };

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${location.host}/api/v1/ws/voice/${auth.sessionId}?token=${encodeURIComponent(auth.token)}`;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    call.ws = ws;

    ws.onopen = () => {
      call.captureNode.port.onmessage = (event) => {
        // The capture worklet posts raw audio (ArrayBuffer) plus occasional
        // diagnostic objects — only audio goes on the wire.
        if (!(event.data instanceof ArrayBuffer)) {
          if (event.data && event.data.type === "gate_stats") {
            const s = event.data;
            // A persistently high gatedRatio means the noise gate is
            // swallowing speech — the failure mode where the agent looks
            // dead because the server never receives any words.
            if (s.gatedRatio > 0.95) {
              console.warn("[voice] mic gate closed almost continuously — " +
                "speech may be getting suppressed", s);
            } else {
              console.debug("[voice] gate stats", s);
            }
          }
          return;
        }
        if (ws.readyState === WebSocket.OPEN && !call.muted) {
          ws.send(event.data);
        }
      };
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        call.playbackNode.port.postMessage({ type: "push", buffer: event.data }, [event.data]);
        return;
      }
      handleServerEvent(JSON.parse(event.data));
    };

    ws.onerror = () => {
      setLabel("Connection trouble…");
    };

    ws.onclose = (event) => {
      if (!call.active) return;
      if (event.code === 4503) {
        setLabel("Voice isn't set up on this server yet.");
      } else if (event.code === 4401) {
        setLabel("Session expired.");
      } else if (event.code !== 1000) {
        setLabel("Call disconnected.");
      }
      setTimeout(endCall, 1800);
    };
  }

  function handleServerEvent(data) {
    switch (data.type) {
      case "ready":
        setOrbState("listening");
        setLabel("Listening…");
        break;

      case "partial_transcript":
        setTranscript(data.text);
        break;

      case "final_transcript":
        setTranscript(data.text);
        setOrbState("thinking");
        setLabel("Thinking…");
        break;

      case "meta":
        if (typeof data.risk_score === "number" && data.risk_score >= 8) setRisk("HIGH");
        break;

      case "speaking_started":
        // Deliberately NOT setOrbState("speaking") here. This fires the
        // instant the server decides to respond — before RAG, generation,
        // or TTS synthesis has produced a single byte. Setting "speaking"
        // this early is exactly what caused "see speaking but no voice is
        // coming": the server orb state jumped to speaking immediately,
        // while the worklet's playback_active (the only signal tied to
        // audio actually reaching the speakers, see audio-playback.js) took
        // however long generation+TTS+network+buffering actually needed —
        // 300ms typically, but several seconds under load, per the "Turn
        // timing" logs. The label updates as an anticipatory hint; the
        // orb only flips once playback_active (line ~133) fires for real.
        setLabel("Speaking…");
        break;

      case "speaking_ended":
        // Same reasoning in reverse: the server has finished SENDING TTS
        // text, but the client may still have buffered audio not yet
        // played. Forcing "listening" here could flip the orb while sound
        // is still coming out of the speakers. playback_idle (fired only
        // once the worklet's queue truly drains) is the truthful signal —
        // this only updates the label and clears the transcript.
        setLabel("Listening…");
        setTranscript("");
        break;

      // Stage 1 of barge-in: mic energy suggests the user started talking.
      // Stop output immediately so we're not talking over them, but keep the
      // queued audio in case it was only noise.
      case "duck":
        call.playbackNode.port.postMessage({ type: "pause" });
        break;

      // Stage 1 turned out to be noise, not speech — carry on where we left off.
      case "unduck":
        call.playbackNode.port.postMessage({ type: "resume" });
        break;

      // Stage 2: real words confirmed. Discard the rest of the reply.
      case "interrupted":
        call.playbackNode.port.postMessage({ type: "clear" });
        setOrbState("listening");
        setLabel("Listening…");
        break;

      case "done":
        setRisk(data.risk_level || "LOW");
        break;

      case "stt_reconnecting":
        // Transient and self-healing (the server reconnects within ~20ms of
        // the next audio packet) — say "reconnecting", not "error", so a
        // blip that has already recovered doesn't read as a dead call.
        setLabel("Reconnecting…");
        break;

      case "error":
        setLabel(data.message || "Something went wrong.");
        break;
    }
  }

  function startLevelMeter() {
    const buf = new Uint8Array(call.analyser.fftSize);
    const tick = () => {
      if (!call.active) return;
      call.analyser.getByteTimeDomainData(buf);
      let sumSquares = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sumSquares += v * v;
      }
      const level = Math.min(1, Math.sqrt(sumSquares / buf.length) * 4);
      orb.style.setProperty("--voice-level", level.toFixed(3));
      call.levelRAF = requestAnimationFrame(tick);
    };
    tick();
  }

  function toggleMute() {
    call.muted = !call.muted;
    muteBtn.setAttribute("aria-pressed", String(call.muted));
    muteBtn.textContent = call.muted ? "🔇" : "🎙";
    if (call.micStream) {
      call.micStream.getAudioTracks().forEach((t) => { t.enabled = !call.muted; });
    }
  }

  function endCall() {
    call.active = false;

    if (call.levelRAF) cancelAnimationFrame(call.levelRAF);
    if (call.ws) {
      call.ws.onclose = null;
      call.ws.close(1000, "user ended call");
      call.ws = null;
    }
    if (call.micStream) {
      call.micStream.getTracks().forEach((t) => t.stop());
      call.micStream = null;
    }
    if (call.captureCtx) { call.captureCtx.close(); call.captureCtx = null; }
    if (call.playbackCtx) { call.playbackCtx.close(); call.playbackCtx = null; }
    call.captureNode = null;
    call.playbackNode = null;
    call.analyser = null;
    call.muted = false;
    muteBtn.setAttribute("aria-pressed", "false");
    muteBtn.textContent = "🎙";

    voiceScreen.hidden = true;
    chatScreen.hidden = false;
  }

  // Voice call needs AudioWorklet + getUserMedia — hide the entry point if unsupported.
  if (!window.AudioWorklet || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    callBtn.hidden = true;
  }
})();
