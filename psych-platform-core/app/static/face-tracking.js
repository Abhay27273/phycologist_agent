/* ============================================================
   Haven — real-time facial expression capture (Phase 1 of the VC feature).

   Uses MediaPipe Face Landmarker (Google, on-device, free, no API key —
   runs entirely client-side via WASM) to extract 52 ARKit-style blendshape
   scores from the user's webcam in real time.

   Phase 1 scope, deliberately: prove capture works and surface it visibly.
   Nothing here talks to the server yet, and nothing here drives an avatar
   yet — there isn't one built. See the phased plan: this validates the
   riskiest new input (webcam -> usable expression signal) before any
   avatar-rendering work is built on top of it.

   Exposed as window.HavenFaceTracking rather than an ES module export
   consumed elsewhere, so voice-call.js (a plain, non-module script,
   matching the rest of this codebase's no-build-step approach) can call
   into it without itself needing to become a module. This file is the only
   one loaded with type="module" (see index.html), since MediaPipe's
   tasks-vision package is only distributed as an ES module.
   ============================================================ */

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";

// Pinned version, not @latest — verified against the npm registry
// (1.0.1 was current as of writing). An unannounced upstream release
// changing behavior underneath a live voice call is exactly the kind of
// surprise this project has spent a lot of effort eliminating elsewhere
// (see the LLM provider circuit-breaker work). Bump deliberately.
const TASKS_VISION_VERSION = "1.0.1";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
// Google's own publicly-hosted model file — the same one their official
// web demos load directly, no self-hosting needed for this phase.
// Verified reachable and a valid .task file before wiring this in.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const state = {
  landmarker: null,
  stream: null,
  videoEl: null,
  rafId: null,
  onResults: null,
  lastVideoTime: -1,
};

async function loadLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU", // falls back to CPU automatically on devices that can't GPU-delegate
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false, // not needed until an avatar exists (Phase 3+)
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

function detectLoop() {
  if (!state.landmarker || !state.videoEl) return;
  const video = state.videoEl;

  // detectForVideo requires a strictly-new frame each call (it throws on a
  // repeated timestamp) — currentTime only advances when a new frame has
  // actually decoded, so this also naturally paces detection to the
  // camera's real frame rate instead of spinning uselessly faster.
  if (video.currentTime !== state.lastVideoTime && video.readyState >= 2) {
    state.lastVideoTime = video.currentTime;
    const results = state.landmarker.detectForVideo(video, performance.now());
    if (state.onResults && results.faceBlendshapes && results.faceBlendshapes.length > 0) {
      state.onResults(results.faceBlendshapes[0].categories);
    }
  }
  state.rafId = requestAnimationFrame(detectLoop);
}

/**
 * Starts webcam capture + real-time face landmark/blendshape detection.
 *
 * @param {HTMLVideoElement} videoEl - existing <video> element to attach
 *   the camera stream to (used as both the self-view and the model's input).
 * @param {(categories: {categoryName: string, score: number}[]) => void} onResults
 *   - called every frame a face is detected; 52 ARKit-style blendshapes.
 * @param {(err: Error) => void} onError - called if camera access or model
 *   loading fails. Voice calls must keep working without this — a
 *   face-tracking failure must never be fatal to the call itself, so this
 *   is always a soft failure reported upward, never a thrown exception.
 */
async function start(videoEl, onResults, onError) {
  state.videoEl = videoEl;
  state.onResults = onResults;

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: "user" },
    });
  } catch (err) {
    onError && onError(new Error("Camera access denied or unavailable: " + err.message));
    return;
  }

  videoEl.srcObject = state.stream;
  videoEl.hidden = false;
  try {
    await videoEl.play();
  } catch (err) {
    onError && onError(new Error("Camera preview failed to start: " + err.message));
    stop();
    return;
  }

  try {
    state.landmarker = await loadLandmarker();
  } catch (err) {
    onError && onError(new Error("Face landmark model failed to load: " + err.message));
    stop();
    return;
  }

  detectLoop();
}

function stop() {
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.rafId = null;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  if (state.videoEl) {
    state.videoEl.srcObject = null;
    state.videoEl.hidden = true;
  }
  if (state.landmarker) {
    state.landmarker.close();
    state.landmarker = null;
  }
  state.lastVideoTime = -1;
}

window.HavenFaceTracking = { start, stop };
