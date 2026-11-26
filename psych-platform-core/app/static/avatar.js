/* ============================================================
   Haven — VRM avatar rendering (Three.js + @pixiv/three-vrm).

   VC feature plan:
   - Phase 3: load and render a VRM avatar (scene/camera/lighting/loader
     pipeline, static bust framing off each model's real bounding box).
   - Phase 4: mouth-open amount driven by TTS playback loudness — see
     setMouthOpen() and video-call.js's startLevelMeter().
   - Phase 5: mood -> VRM expression preset, smoothly blended — see
     setMoodExpression() and MOOD_TO_EXPRESSION below.
   Phase 6 (integration polish — an in-call way to set the avatar URL
   without devtools, plus closing the gaze_avoidance signal gap) lives in
   video-call.js, not here.

   Originally planned around Ready Player Me, but RPM was acquired by
   Netflix and fully shut down (avatar creator, PlayerZero, and all
   public APIs offline) on 2026-01-31 — confirmed live, not assumed, since
   readyplayer.me no longer resolves at all. VRoid Studio (free, actively
   maintained) + the VRM format is the replacement: create an avatar at
   https://vroid.com/studio, export a .vrm file, host it anywhere reachable
   over HTTPS, then call HavenAvatar.setAvatarUrl(url) from the console —
   it's saved to localStorage so it persists across calls. Until a URL is
   set, a plain placeholder bust renders instead of a blank canvas, so the
   pipeline is visibly alive either way.

   Exposes window.HavenAvatar so video-call.js (a plain script, not a
   module) can drive it — same pattern face-tracking.js already uses for
   window.HavenFaceTracking.

   VRM avatars vary in height/proportions far more than Ready Player Me's
   fairly standardized rig did (VRoid presets alone range roughly
   1.4m-1.7m), so the camera is framed dynamically off each loaded model's
   actual bounding box rather than a hardcoded head height — see
   frameCameraOnBust().
   ============================================================ */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const AVATAR_URL_STORAGE_KEY = "haven_avatar_url";

// VC feature, Phase 5: mood -> VRM expression preset. Deliberately NOT a
// mirror of the user's own mood (confirmed design decision from Phase 1
// planning) — e.g. "angry" maps to a steady "relaxed" response, not the
// avatar getting angry back. "grieving" is the one deliberate exception:
// a little shared, gentle sadness reads as validation rather than as
// mismatched cheerfulness. Weights are a starting point, not calibrated
// against any user testing.
const MOOD_TO_EXPRESSION = {
  anxious: { name: "relaxed", weight: 0.6 },
  depressed: { name: "sad", weight: 0.45 },
  lonely: { name: "sad", weight: 0.4 },
  angry: { name: "relaxed", weight: 0.6 },
  stressed: { name: "relaxed", weight: 0.55 },
  fearful: { name: "relaxed", weight: 0.6 },
  hopeless: { name: "sad", weight: 0.4 },
  guilty: { name: "relaxed", weight: 0.5 },
  confused: { name: "neutral", weight: 0 },
  traumatized: { name: "relaxed", weight: 0.65 },
  grieving: { name: "sad", weight: 0.55 },
  calm: { name: "happy", weight: 0.3 },
  neutral: { name: "neutral", weight: 0 },
  somatic: { name: "relaxed", weight: 0.5 },
  tension: { name: "relaxed", weight: 0.55 },
  anhedonic: { name: "sad", weight: 0.35 },
};

// Every VRM expression name this module ever sets — tracked so a
// no-longer-targeted expression gets explicitly lerped back to 0 rather
// than left stuck at its last value.
const MANAGED_EXPRESSION_NAMES = ["happy", "sad", "angry", "relaxed", "surprised", "neutral", "aa"];

const state = {
  renderer: null,
  scene: null,
  camera: null,
  canvas: null,
  avatarRoot: null, // whichever Object3D is currently displayed
  currentVrm: null, // set once a real VRM loads; null while showing the placeholder
  rafId: null,
  lastFrameTime: null,
  loader: null,
  // VC feature, Phase 4/5: current + target blend weights, lerped once per
  // frame in renderLoop() rather than snapped, so mood/mouth changes read
  // as smooth transitions instead of jarring flips between frames.
  expressionWeights: {}, // name -> currently-applied weight (0-1)
  moodTarget: { name: "neutral", weight: 0 },
  mouthTarget: 0, // 0-1, driven by TTS playback loudness — see setMouthOpen()
};

function buildScene(canvas) {
  const scene = new THREE.Scene();

  // Generic starting pose — frameCameraOnBust() repositions this as soon
  // as any root (placeholder or real avatar) is set, so these numbers
  // only matter for the single frame before that happens.
  const camera = new THREE.PerspectiveCamera(30, canvas.clientWidth / canvas.clientHeight, 0.1, 10);
  camera.position.set(0, 1.4, 1);
  camera.lookAt(0, 1.4, 0);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene.add(new THREE.AmbientLight(0xffffff, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(0.6, 1.8, 1.2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5);
  fill.position.set(-0.8, 1.2, 0.6);
  scene.add(fill);

  return { scene, camera, renderer };
}

function buildLoader() {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  return loader;
}

function buildPlaceholder() {
  // Stand-in bust shown until a real avatar URL is configured, so the
  // canvas is never just blank — this alone proves the render loop,
  // camera, and lighting work independent of having a VRM file on hand.
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color: 0x8a7cff, roughness: 0.55, metalness: 0.05 });
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 32, 32), material);
  head.position.set(0, 1.62, 0);
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.16, 0.32, 8, 16), material);
  torso.position.set(0, 1.32, 0);
  group.add(head, torso);
  return group;
}

// Frames a bust shot (head + shoulders) off whatever the model's actual
// size turns out to be, instead of trusting a fixed head-height constant
// — necessary now that avatars aren't all the same RPM-standardized rig.
function frameCameraOnBust(root) {
  if (!state.camera) return;
  const box = new THREE.Box3().setFromObject(root);
  const height = box.max.y - box.min.y;
  if (!isFinite(height) || height <= 0) return; // empty/degenerate bbox — keep current framing
  const headY = box.min.y + height * 0.9; // just below the crown
  const distance = Math.max(height * 0.32, 0.4);
  state.camera.position.set(0, headY, distance);
  state.camera.lookAt(0, headY - height * 0.02, 0);
}

function resize() {
  if (!state.renderer || !state.canvas) return;
  const { clientWidth, clientHeight } = state.canvas;
  if (!clientWidth || !clientHeight) return;
  state.camera.aspect = clientWidth / clientHeight;
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(clientWidth, clientHeight, false);
}

// Exponential-ish approach toward `target` — frame-rate independent
// (unlike a flat per-frame step) since it's scaled by delta.
function lerpTowards(current, target, delta, speedPerSecond) {
  const t = Math.min(1, delta * speedPerSecond);
  return current + (target - current) * t;
}

function applyExpressionBlending(delta) {
  if (!state.currentVrm?.expressionManager) return;
  const manager = state.currentVrm.expressionManager;
  for (const name of MANAGED_EXPRESSION_NAMES) {
    let target = 0;
    if (name === "aa") {
      target = state.mouthTarget;
    } else if (name === state.moodTarget.name) {
      target = state.moodTarget.weight;
    }
    // Mouth movement needs to visibly track loudness in near-real-time;
    // mood is meant to read as a slow, calm shift, not a flicker — hence
    // the very different speeds for the same lerp.
    const speed = name === "aa" ? 14 : 3;
    const current = state.expressionWeights[name] || 0;
    const next = lerpTowards(current, target, delta, speed);
    state.expressionWeights[name] = next;
    manager.setValue(name, next);
  }
}

function renderLoop(time) {
  state.rafId = requestAnimationFrame(renderLoop);
  const delta = state.lastFrameTime != null ? (time - state.lastFrameTime) / 1000 : 0;
  state.lastFrameTime = time;
  applyExpressionBlending(delta);
  // VRM spring-bones/look-at/expressions all need this per frame — it's
  // what actually turns expressionManager.setValue() above into morph
  // target changes on the mesh.
  if (state.currentVrm) state.currentVrm.update(delta);
  state.renderer.render(state.scene, state.camera);
}

// VC feature, Phase 4: mouth-open amount (0-1) driven by TTS playback
// loudness — see video-call.js's startLevelMeter(). This is loudness, not
// real lip-sync: without phoneme/viseme timing from the TTS engine there's
// no honest basis to distinguish "aa" from "ih"/"ou"/"ee"/"oh", so this
// only ever drives a generic mouth-open shape, not specific vowel shapes.
function setMouthOpen(amount) {
  state.mouthTarget = Math.max(0, Math.min(1, amount));
}

// VC feature, Phase 5: mood string (one of sentiment.py's _VALID_MOODS) ->
// target VRM expression. Unknown moods fall back to neutral rather than
// throwing, since the mood vocabulary lives server-side and could grow.
function setMoodExpression(mood) {
  state.moodTarget = MOOD_TO_EXPRESSION[mood] || { name: "neutral", weight: 0 };
}

function replaceAvatarRoot(newRoot) {
  if (state.avatarRoot) state.scene.remove(state.avatarRoot);
  state.avatarRoot = newRoot;
  state.scene.add(newRoot);
  frameCameraOnBust(newRoot);
}

async function init(canvas) {
  state.canvas = canvas;
  const built = buildScene(canvas);
  state.scene = built.scene;
  state.camera = built.camera;
  state.renderer = built.renderer;
  state.loader = buildLoader();
  state.currentVrm = null;
  state.lastFrameTime = null;
  state.expressionWeights = {};
  state.moodTarget = { name: "neutral", weight: 0 };
  state.mouthTarget = 0;

  replaceAvatarRoot(buildPlaceholder());
  resize();
  window.addEventListener("resize", resize);
  renderLoop(performance.now());

  const savedUrl = localStorage.getItem(AVATAR_URL_STORAGE_KEY);
  if (savedUrl) await setAvatarUrl(savedUrl);
}

async function setAvatarUrl(url) {
  if (!url || !state.loader) return;
  try {
    const gltf = await state.loader.loadAsync(url);
    const vrm = gltf.userData.vrm;
    if (!vrm) throw new Error("loaded glTF has no VRM data — is this actually a .vrm file?");

    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    VRMUtils.combineMorphs(vrm);
    // VRM 0.x models face -Z (backwards, relative to the VRM 1.0 spec's
    // +Z) — this is the documented three-vrm fix, not a guess; VRoid
    // Studio still exports 0.x by default as of this writing.
    if (vrm.meta?.metaVersion === "0") VRMUtils.rotateVRM0(vrm);

    replaceAvatarRoot(vrm.scene);
    state.currentVrm = vrm;
    localStorage.setItem(AVATAR_URL_STORAGE_KEY, url);
  } catch (err) {
    console.warn("HavenAvatar: failed to load VRM avatar, keeping placeholder", err);
  }
}

function dispose() {
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.rafId = null;
  window.removeEventListener("resize", resize);
  if (state.renderer) state.renderer.dispose();
  state.renderer = null;
  state.scene = null;
  state.camera = null;
  state.canvas = null;
  state.avatarRoot = null;
  state.currentVrm = null;
  state.loader = null;
  state.lastFrameTime = null;
  state.expressionWeights = {};
  state.moodTarget = { name: "neutral", weight: 0 };
  state.mouthTarget = 0;
}

window.HavenAvatar = { init, setAvatarUrl, dispose, setMouthOpen, setMoodExpression };
