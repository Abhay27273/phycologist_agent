/* ============================================================
   Haven — real-time video call orchestration.

   A deliberately separate feature from voice-call.js, not voice-call.js
   plus a bolted-on camera — its own entry point (video-call-btn), its own
   screen (video-call-screen), its own file. The two share the same backend
   voice pipeline (STT -> sentiment -> strategy -> generation -> TTS, same
   /ws/voice/{session_id} endpoint) since that doesn't change; what's
   actually new here is entirely client-side: webcam capture via
   face-tracking.js (window.HavenFaceTracking).

   The audio plumbing below (two AudioContexts, AudioWorklets, WS framing,
   barge-in handling) intentionally mirrors voice-call.js line-for-line
   rather than importing/sharing it — voice-call.js has been through a lot
   of hard-won debugging this session (turn-detection races, barge-in
   staging, orb-state timing) and refactoring it into a shared module right
   now would risk regressing an already-correct, already-verified feature
   for the sake of avoiding duplication. Revisit sharing the two once
   video-call.js is equally battle-tested.

   VC feature plan, current phase: Phase 6 (integration polish) — facial
   capture, nonverbal-cues channel, VRM avatar rendering, mouth-sync, and
   mood-driven expressions (Phases 1-5) are all wired in; this phase adds
   an in-call way to set the avatar URL (changeAvatar, below) and gaze
   detection in summarizeFaceCategories, so nothing here requires devtools.
   ============================================================ */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const chatScreen = $("chat-screen");
  const videoCallScreen = $("video-call-screen");
  const videoCallBtn = $("video-call-btn");
  const endCallBtn = $("video-call-end-btn");
  const muteBtn = $("video-call-mute-btn");
  const avatarBtn = $("video-call-avatar-btn");
  const orb = $("video-call-orb");
  const stateLabel = $("video-call-state-label");
  const transcriptEl = $("video-call-transcript");
  const riskBadge = $("video-call-risk-badge");
  const faceVideo = $("face-track-video");
  const faceDebug = $("face-track-debug");
  const avatarCanvas = $("video-call-avatar-canvas");

  const call = {
    ws: null,
    captureCtx: null,
    playbackCtx: null,
    captureNode: null,
    playbackNode: null,
    analyser: null,
    playbackAnalyser: null, // VC feature, Phase 4 — TTS playback loudness tap
    micStream: null,
    levelRAF: null,
    muted: false,
    active: false,
    // VC feature, Phase 2.
    toneSamples: [],           // rolling {pitch, energy} samples, drained every send interval
    latestFaceCategories: null, // most recent MediaPipe blendshape array, or null
    sendCuesInterval: null,
  };

  function sessionKeyFor(userId) {
    return `haven_session_${userId}`;
  }

  function authState() {
    const token = localStorage.getItem("haven_token");
    const userId = localStorage.getItem("haven_user_id");
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

  videoCallBtn.addEventListener("click", startCall);
  endCallBtn.addEventListener("click", endCall);
  muteBtn.addEventListener("click", toggleMute);
  avatarBtn.addEventListener("click", changeAvatar);

  // VC feature, Phase 6: a plain `prompt()` rather than a custom modal —
  // setAvatarUrl() previously had no reachable UI at all (console-only),
  // which meant the feature only ever showed the geometric placeholder bust
  // to anyone who hadn't opened devtools. A native prompt is unglamorous
  // but gets a real VRM avatar on screen without adding a modal/build step
  // to a project that deliberately has neither. Persisted avatar URL is
  // already handled inside HavenAvatar (localStorage), so this only needs
  // to collect the URL and hand it off.
  function changeAvatar() {
    if (!window.HavenAvatar) return;
    const current = localStorage.getItem("haven_avatar_url") || "";
    const url = window.prompt(
      "VRM avatar URL (export one free at https://vroid.com/studio):",
      current
    );
    if (url) window.HavenAvatar.setAvatarUrl(url.trim());
  }

  async function startCall() {
    const auth = authState();
    if (!auth) return;

    chatScreen.hidden = true;
    videoCallScreen.hidden = false;
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
      setLabel("Microphone access is required for a video call.");
      setTimeout(endCall, 2500);
      return;
    }

    // Facial capture — fire-and-forget, not awaited: an optional layer on
    // top of the call, not a dependency of it. Camera permission prompts
    // right after the mic prompt above rather than appearing later mid-call.
    // A denied/unsupported camera (including window.HavenFaceTracking never
    // having loaded at all — ad blocker, CDN blocked, no ES module support)
    // must never affect the call itself; see startFaceTracking's guards.
    startFaceTracking();

    // VC feature, Phase 3: same fire-and-forget treatment as facial capture
    // above — an optional visual layer, never a dependency of the call.
    startAvatar();

    // VC feature, Phase 2: every 2s, summarize the rolling voice-tone
    // samples + latest facial blendshapes and send them to the server.
    // 2s, not per-frame: sentiment analysis runs per-turn, not per-frame —
    // there's no value in updating faster than the thing that consumes it,
    // and sendNonverbalCues already no-ops if there's nothing worth sending.
    call.toneSamples = [];
    call.sendCuesInterval = setInterval(sendNonverbalCues, 2000);

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

    // VC feature, Phase 4: tap the TTS playback signal (same pattern as
    // the mic analyser above) so startLevelMeter() can drive the avatar's
    // mouth-open amount from actual reply loudness, not a guessed timer.
    call.playbackAnalyser = call.playbackCtx.createAnalyser();
    call.playbackAnalyser.fftSize = 512;
    call.playbackNode.connect(call.playbackAnalyser);

    call.playbackNode.port.onmessage = (event) => {
      if (event.data.type === "playback_active") setOrbState("speaking");
      else if (event.data.type === "playback_idle" && call.active) setOrbState("listening");
    };

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${location.host}/api/v1/ws/voice/${auth.sessionId}?token=${encodeURIComponent(auth.token)}&lang=${encodeURIComponent(getCallLanguage())}`;
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
              console.warn("[video-call] mic gate closed almost continuously — " +
                "speech may be getting suppressed", s);
            } else {
              console.debug("[video-call] gate stats", s);
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
        // VC feature, Phase 5: react to the user's detected mood, not
        // mirror it — see MOOD_TO_EXPRESSION in avatar.js.
        if (data.mood && window.HavenAvatar) window.HavenAvatar.setMoodExpression(data.mood);
        break;

      case "speaking_started":
        // Deliberately NOT setOrbState("speaking") here — see voice-call.js
        // for the full reasoning (the "speaking shown but no audio yet"
        // bug this avoids). The orb only flips once playback_active fires.
        setLabel("Speaking…");
        break;

      case "speaking_ended":
        setLabel("Listening…");
        setTranscript("");
        break;

      // Stage 1 of barge-in: mic energy suggests the user started talking.
      case "duck":
        call.playbackNode.port.postMessage({ type: "pause" });
        break;

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
        setLabel("Reconnecting…");
        break;

      case "error":
        setLabel(data.message || "Something went wrong.");
        break;
    }
  }

  function startLevelMeter() {
    const buf = new Float32Array(call.analyser.fftSize);
    let playbackBuf = null; // allocated lazily — playbackAnalyser doesn't exist yet at this call site
    let lastPitchSample = 0;
    const tick = () => {
      if (!call.active) return;
      call.analyser.getFloatTimeDomainData(buf);
      let sumSquares = 0;
      for (let i = 0; i < buf.length; i++) sumSquares += buf[i] * buf[i];
      const rms = Math.sqrt(sumSquares / buf.length);
      const level = Math.min(1, rms * 4);
      orb.style.setProperty("--voice-level", level.toFixed(3));

      // VC feature, Phase 4: drive the avatar's mouth from actual TTS
      // playback loudness — an amplitude proxy, not real viseme lip-sync
      // (see avatar.js's setMouthOpen for why). *5 is an empirical scale
      // picked by ear, not calibrated.
      if (call.playbackAnalyser && window.HavenAvatar) {
        if (!playbackBuf || playbackBuf.length !== call.playbackAnalyser.fftSize) {
          playbackBuf = new Float32Array(call.playbackAnalyser.fftSize);
        }
        call.playbackAnalyser.getFloatTimeDomainData(playbackBuf);
        let playbackSumSquares = 0;
        for (let i = 0; i < playbackBuf.length; i++) playbackSumSquares += playbackBuf[i] * playbackBuf[i];
        const playbackRms = Math.sqrt(playbackSumSquares / playbackBuf.length);
        window.HavenAvatar.setMouthOpen(playbackRms * 5);
      }

      // VC feature, Phase 2: sample pitch at ~10Hz, not every frame — the
      // autocorrelation below is O(n^2) over the buffer and doesn't need
      // 60fps to capture how pitch drifts/wavers over a multi-second
      // window; this just cuts the CPU cost with no loss to the heuristic.
      const now = performance.now();
      if (now - lastPitchSample >= 100) {
        lastPitchSample = now;
        const pitch = autoCorrelate(buf, call.captureCtx.sampleRate);
        call.toneSamples.push({ pitch, energy: rms });
      }

      call.levelRAF = requestAnimationFrame(tick);
    };
    tick();
  }

  // Standard autocorrelation pitch (F0) estimator — the same well-known
  // technique used in browser-based tuner/pitch-detection demos, not a
  // novel algorithm invented for this project. Returns Hz, or -1 when the
  // signal is too quiet or outside plausible human vocal range (an
  // autocorrelation octave error/noise artifact, not real pitch) —
  // callers must treat -1 as "no reading this sample", not silence/zero.
  function autoCorrelate(buf, sampleRate) {
    const SIZE = buf.length;
    let rms = 0;
    for (let i = 0; i < SIZE; i++) rms += buf[i] * buf[i];
    rms = Math.sqrt(rms / SIZE);
    if (rms < 0.01) return -1; // too quiet for a reliable estimate

    let r1 = 0, r2 = SIZE - 1;
    const threshold = 0.2;
    for (let i = 0; i < SIZE / 2; i++) {
      if (Math.abs(buf[i]) < threshold) { r1 = i; break; }
    }
    for (let i = 1; i < SIZE / 2; i++) {
      if (Math.abs(buf[SIZE - i]) < threshold) { r2 = SIZE - i; break; }
    }
    const trimmed = buf.slice(r1, r2);
    const newSize = trimmed.length;
    if (newSize < 8) return -1;

    const c = new Float32Array(newSize);
    for (let lag = 0; lag < newSize; lag++) {
      let sum = 0;
      for (let i = 0; i < newSize - lag; i++) sum += trimmed[i] * trimmed[i + lag];
      c[lag] = sum;
    }

    let d = 0;
    while (d < newSize - 1 && c[d] > c[d + 1]) d++;
    let maxVal = -1, maxPos = -1;
    for (let i = d; i < newSize; i++) {
      if (c[i] > maxVal) { maxVal = c[i]; maxPos = i; }
    }
    if (maxPos <= 0) return -1;
    let T0 = maxPos;

    // Parabolic interpolation around the peak for sub-sample precision.
    const x1 = T0 > 0 ? c[T0 - 1] : 0;
    const x2 = c[T0];
    const x3 = T0 + 1 < newSize ? c[T0 + 1] : 0;
    const a = (x1 + x3 - 2 * x2) / 2;
    const b = (x3 - x1) / 2;
    if (a !== 0) T0 = T0 - b / (2 * a);
    if (T0 <= 0) return -1;

    const freq = sampleRate / T0;
    if (freq < 60 || freq > 500) return -1; // outside plausible human vocal range
    return freq;
  }

  // ---------------------------------------------------------
  // VC feature, Phase 2: voice-tone + facial cues -> server.
  //
  // A deliberately conservative heuristic, not a certain classifier —
  // reliably detecting something like "crying" from voice alone isn't a
  // solved, off-the-shelf problem. This surfaces a distress SIGNAL (pitch
  // instability + energy pattern) and only sets a tone/emotion label when
  // the signal is actually distinctive, leaving fields unset otherwise
  // rather than forcing a guess every 2 seconds. See _build_multimodal_hint
  // in sentiment.py for exactly what shape these feed into.
  // ---------------------------------------------------------

  function summarizeToneSamples() {
    const samples = call.toneSamples;
    call.toneSamples = [];
    if (samples.length === 0) return null;

    const energies = samples.map((s) => s.energy);
    const meanEnergy = energies.reduce((a, b) => a + b, 0) / energies.length;

    const pitches = samples.map((s) => s.pitch).filter((p) => p > 0);
    let tone = null;
    if (pitches.length >= 3) {
      const meanPitch = pitches.reduce((a, b) => a + b, 0) / pitches.length;
      const variance =
        pitches.reduce((a, p) => a + (p - meanPitch) ** 2, 0) / pitches.length;
      const stdDev = Math.sqrt(variance);
      // Thresholds are a starting heuristic, not calibrated against real
      // labeled distress audio — expect to tune these once there's real
      // usage to look at, the same way sentiment calibration was tuned
      // against test fixtures earlier this project.
      const coefficientOfVariation = stdDev / meanPitch;
      if (coefficientOfVariation > 0.25 && meanEnergy > 0.1) {
        tone = "shaky/unstable pitch";
      } else if (meanEnergy < 0.05) {
        tone = "flat/subdued";
      }
    }

    return {
      energy: Math.min(1, meanEnergy * 4),
      ...(tone ? { tone } : {}),
    };
  }

  // Small, deliberately conservative blendshape -> label mapping. Only
  // fires on a clear, high-confidence combination; silent otherwise rather
  // than forcing a guess (same reasoning as the voice-tone heuristic above).
  function summarizeFaceCategories() {
    const categories = call.latestFaceCategories;
    if (!categories) return null;

    const score = (name) => (categories.find((c) => c.categoryName === name) || {}).score || 0;
    const smile = Math.max(score("mouthSmileLeft"), score("mouthSmileRight"));
    const frown = Math.max(score("mouthFrownLeft"), score("mouthFrownRight"));
    const browDown = Math.max(score("browDownLeft"), score("browDownRight"));
    const browUp = Math.max(score("browOuterUpLeft"), score("browOuterUpRight"));
    const eyeWide = Math.max(score("eyeWideLeft"), score("eyeWideRight"));
    const jawOpen = score("jawOpen");

    let dominantEmotion = null;
    if (smile > 0.5) dominantEmotion = "smiling";
    else if (frown > 0.4 && browDown > 0.4) dominantEmotion = "sad/concerned";
    else if (browUp > 0.4 && eyeWide > 0.4) dominantEmotion = "surprised";

    // VC feature, Phase 6: eyes cast down or turned away from the camera —
    // the webcam-equivalent of averted eye contact. Same conservative
    // threshold-only-when-clear approach as the rest of this function: a
    // single ambiguous frame (e.g. glancing at the transcript) should not
    // flag this, so this only fires on the blendshapes most specific to
    // down/sideways gaze, not general eye movement.
    const eyeDown = Math.max(score("eyeLookDownLeft"), score("eyeLookDownRight"));
    const eyeOut = Math.max(score("eyeLookOutLeft"), score("eyeLookOutRight"));
    const gazeAvoidance = eyeDown > 0.5 || eyeOut > 0.6;

    // action_units: pass through MediaPipe's own top active blendshape
    // names directly, rather than inventing a relabeling — these already
    // closely correspond to FACS action units, so relaying them as-is is
    // more honest than a heuristic re-interpretation.
    const actionUnits = [...categories]
      .filter((c) => c.categoryName !== "_neutral" && c.score > 0.3)
      .sort((a, b) => b.score - a.score)
      .slice(0, 4)
      .map((c) => c.categoryName);

    if (!dominantEmotion && !gazeAvoidance && actionUnits.length === 0) return null;
    return {
      ...(dominantEmotion ? { dominant_emotion: dominantEmotion } : {}),
      ...(gazeAvoidance ? { gaze_avoidance: true } : {}),
      ...(actionUnits.length ? { action_units: actionUnits } : {}),
    };
  }

  function sendNonverbalCues() {
    if (!call.ws || call.ws.readyState !== WebSocket.OPEN) return;
    const audio = summarizeToneSamples();
    const video = summarizeFaceCategories();
    if (!audio && !video) return; // nothing worth sending this window
    call.ws.send(JSON.stringify({
      type: "nonverbal_cues",
      ...(audio ? { audio } : {}),
      ...(video ? { video } : {}),
    }));
  }

  // ---------------------------------------------------------
  // VC feature, Phase 1: facial capture (proof-of-capture only —
  // no avatar exists yet; this just proves the input side works).
  // ---------------------------------------------------------

  let lastDebugPaint = 0;

  async function startAvatar() {
    if (!window.HavenAvatar || !avatarCanvas) {
      console.warn("[video-call] avatar rendering unavailable (HavenAvatar not loaded)");
      return;
    }
    try {
      await window.HavenAvatar.init(avatarCanvas);
    } catch (err) {
      console.warn("[video-call] avatar init failed:", err);
    }
  }

  async function startFaceTracking() {
    if (!window.HavenFaceTracking) {
      console.warn("[video-call] face tracking unavailable (HavenFaceTracking not loaded)");
      return;
    }
    await window.HavenFaceTracking.start(
      faceVideo,
      (categories) => {
        if (!call.active) return;
        // Phase 2 reads this on its own ~2s interval (sendNonverbalCues),
        // independent of the debug-paint throttle below — always kept
        // current regardless of whether the debug panel repaints this frame.
        call.latestFaceCategories = categories;
        // Throttled to ~6fps — the detector itself still runs at full
        // camera frame rate (see face-tracking.js); this only limits how
        // often the debug text repaints, since nothing needs 60fps DOM
        // updates for a readout a human is reading.
        const now = performance.now();
        if (now - lastDebugPaint < 160) return;
        lastDebugPaint = now;
        renderFaceDebug(categories);
      },
      (err) => {
        console.warn("[video-call] face tracking disabled:", err.message);
        faceVideo.hidden = true;
        faceDebug.hidden = true;
      }
    );
  }

  function renderFaceDebug(categories) {
    // Top 5 non-neutral blendshapes — "proof it's alive" for Phase 1, not
    // a real UI. categoryName values are MediaPipe's own ARKit-style names
    // (e.g. "mouthSmileLeft", "browDownRight", "jawOpen").
    const top = [...categories]
      .filter((c) => c.categoryName !== "_neutral")
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    faceDebug.hidden = false;
    faceDebug.innerHTML = top
      .map((c) => `${c.categoryName}: <strong>${c.score.toFixed(2)}</strong>`)
      .join("<br>");
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

    if (window.HavenFaceTracking) window.HavenFaceTracking.stop();
    faceDebug.hidden = true;
    faceDebug.innerHTML = "";
    if (window.HavenAvatar) window.HavenAvatar.dispose();

    if (call.sendCuesInterval) { clearInterval(call.sendCuesInterval); call.sendCuesInterval = null; }
    call.toneSamples = [];
    call.latestFaceCategories = null;

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
    call.playbackAnalyser = null;
    call.muted = false;
    muteBtn.setAttribute("aria-pressed", "false");
    muteBtn.textContent = "🎙";

    videoCallScreen.hidden = true;
    chatScreen.hidden = false;
  }

  // Video call needs AudioWorklet + getUserMedia (audio) at minimum — same
  // hard requirement as voice call. Camera is treated as optional (see
  // startFaceTracking), so its absence doesn't hide this entry point.
  if (!window.AudioWorklet || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    videoCallBtn.hidden = true;
  }
})();
