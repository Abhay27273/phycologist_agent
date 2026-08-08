"""
Smart Turn v3.2 (pipecat-ai/smart-turn-v3) — a standalone, BSD-2-Clause,
audio-based semantic turn-completion classifier. Runs locally via ONNX
Runtime; no network round-trip, no LLM cost, ~12-65ms CPU inference vs
~150-450ms for the Groq-based semantic check in groq_service.py.

Verified empirically against our own voice fixtures before wiring this in
(not just trusted from the model card):
  - A clean, unambiguous "complete" utterance, with >=300ms of trailing
    silence (what's actually present by the time Deepgram's speech_final
    fires — endpointing=300), scored confidently complete (~0.98) across
    300ms-1500ms of trailing silence.
  - BUT a second utterance (the crisis-message fixture) showed real
    non-monotonic instability: confidently complete at 300ms (0.80) and
    1500ms (0.62), but dipped BELOW the model's own suggested 0.5
    threshold at 1000ms specifically (0.28). Reproduced, not a fluke.

Given that, this is used as an ADDITIONAL fast-path signal alongside (not
a replacement for) the existing LLM-based semantic check and Deepgram's
UtteranceEnd fallback in voice.py — with a conservative 0.7 threshold
(vs the model's own 0.5 default) to only act in the confidence band this
model demonstrated to actually be reliable in. Any failure here (model
unavailable, low/uncertain confidence) is treated as "no opinion" and
simply defers to the other signals — this can only make turn-taking
faster, never slower or less safe.
"""
import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_REPO = "pipecat-ai/smart-turn-v3"
MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
COMPLETE_THRESHOLD = 0.7


class SmartTurnDetector:
    """Lazily-loaded singleton — one ONNX session + feature extractor shared
    across all voice calls. Stateless CPU inference, safe to share."""

    def __init__(self):
        self._session = None
        self._feature_extractor = None
        self._lock = asyncio.Lock()
        self._load_failed = False

    async def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self._load_failed:
            return False
        async with self._lock:
            if self._session is not None:
                return True
            if self._load_failed:
                return False
            try:
                await asyncio.to_thread(self._load)
                return True
            except Exception as e:
                logger.error("SmartTurnDetector failed to load — disabling for this process: %s", e)
                self._load_failed = True
                return False

    def _load(self) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_options=so)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=WINDOW_SECONDS)
        logger.info("SmartTurnDetector loaded (%s)", MODEL_FILENAME)

    def _predict_sync(self, audio_float32_16k: np.ndarray) -> float:
        audio = audio_float32_16k
        max_samples = WINDOW_SECONDS * SAMPLE_RATE
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (max_samples - len(audio), 0), mode="constant")

        inputs = self._feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np",
            padding="max_length", max_length=max_samples, truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)
        outputs = self._session.run(None, {"input_features": input_features})
        return float(outputs[0][0].item())

    async def is_utterance_complete(self, audio_int16: bytes) -> Optional[bool]:
        """audio_int16: raw 16-bit mono PCM @ 16kHz — the recent rolling
        audio buffer, not the transcript. Returns True/False, or None if
        the model isn't available/inference failed (caller must treat None
        as "no opinion", never as "incomplete")."""
        if not audio_int16 or not await self._ensure_loaded():
            return None
        audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            probability = await asyncio.to_thread(self._predict_sync, audio_float32)
        except Exception as e:
            logger.error("SmartTurnDetector inference failed: %s", e)
            return None
        return probability > COMPLETE_THRESHOLD


smart_turn_detector = SmartTurnDetector()
