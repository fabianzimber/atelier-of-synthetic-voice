"""
core/engine.py – TTS-Engine (Singleton)

Verwaltet das Qwen3-TTS-Voice-Clone-Modell:
  • Qwen3-TTS-12Hz-1.7B-Base → Voice-Cloning via Referenz-Audio

Apple-Silicon / MPS-Hinweise (macOS 26 + PyTorch 2.10)
-------------------------------------------------------
• torch.bfloat16 ist stabil auf MPS (getestet mit macOS 26 Beta 4).
• attn_implementation="sdpa" (PyTorch Scaled Dot Product Attention).
• torch.mps.synchronize() nach jeder Inferenz aufrufen.
"""

from __future__ import annotations

import warnings
from threading import Lock
from typing import Callable, Optional, Union

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*flash.attn.*")
warnings.filterwarnings("ignore", message=".*FlashAttention.*")

# ── Modell-Identifikation ────────────────────────────────────────────────────
CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# ── Unterstützte Sprachen ────────────────────────────────────────────────────
SUPPORTED_LANGUAGES: list[str] = [
    "Auto",
    "Chinese",
    "English",
    "German",
    "French",
    "Japanese",
    "Korean",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TTSEngine:
    """Thread-sicherer Singleton, der beide TTS-Modelle verwaltet."""

    _instance: Optional["TTSEngine"] = None
    _creation_lock: Lock = Lock()

    # ── Singleton-Konstruktion ─────────────────────────────────────────────
    def __new__(cls) -> "TTSEngine":
        with cls._creation_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._clone_model = None
        self._model_lock = Lock()
        self._device = _detect_device()
        self._initialized = True

    # ── Eigenschaften ──────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        return self._device

    @property
    def device_label(self) -> str:
        labels = {"mps": "Apple Silicon (MPS)", "cuda": "NVIDIA CUDA", "cpu": "CPU"}
        return labels.get(self._device, self._device)

    @property
    def clone_model_loaded(self) -> bool:
        return self._clone_model is not None

    # ── Modell laden ───────────────────────────────────────────────────────
    def load_clone_model(
        self, progress_cb: Optional[Callable[[str], None]] = None
    ) -> object:
        with self._model_lock:
            if self._clone_model is not None:
                return self._clone_model
            if progress_cb:
                progress_cb("Lade Voice-Clone-Modell (1.7B)…")

            from qwen_tts import Qwen3TTSModel  # type: ignore[import]

            self._clone_model = Qwen3TTSModel.from_pretrained(
                CLONE_MODEL_ID,
                device_map={"": self._device},
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            # Speech-Tokenizer landet auf CPU weil from_pretrained()
            # den device_map aus kwargs poppt bevor der Tokenizer geladen wird.
            st = self._clone_model.model.speech_tokenizer
            if st is not None and str(st.device) != self._device:
                if progress_cb:
                    progress_cb("Verschiebe Speech-Tokenizer auf MPS…")
                st.model = st.model.to(self._device)
                st.device = torch.device(self._device)

            # torch.compile für ~3x Speedup auf MPS
            if progress_cb:
                progress_cb("Optimiere Modell mit torch.compile…")
            talker = self._clone_model.model.talker
            talker.model = torch.compile(talker.model, backend="aot_eager")
            talker.code_predictor.model = torch.compile(
                talker.code_predictor.model, backend="aot_eager"
            )

            if progress_cb:
                progress_cb("Voice-Clone-Modell bereit.")
        return self._clone_model

    # ── Inferenz: Voice Cloning ────────────────────────────────────────────
    def generate_with_clone(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str]],
        ref_audio: Union[str, tuple[np.ndarray, int]],
        ref_text: str,
        progress_cb: Optional[Callable[[str], None]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_temperature: float = 0.9,
    ) -> tuple[list[np.ndarray], int]:
        """Generiert Sprache durch direkte Referenz-Audio-Übergabe."""
        model = self.load_clone_model(progress_cb)
        if progress_cb:
            progress_cb("Generiere Sprache mit geklonter Stimme…")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=subtalker_temperature,
        )
        self._sync_device()
        return wavs, sr

    # ── Hilfsmethoden ─────────────────────────────────────────────────────
    def _sync_device(self) -> None:
        if self._device == "mps":
            torch.mps.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()
