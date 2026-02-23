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
        max_new_tokens: int = 4092,
        temperature: float = 1.8,
        repetition_penalty: float = 1.05,
        subtalker_temperature: float = 1.8,
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
            temperature=1.6, #temperature,
            repetition_penalty=1.1, #epetition_penalty,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=1.6, #subtalker_temperature,
        )
        self._sync_device()
        return wavs, sr

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple[np.ndarray, int]],
        ref_text: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> object:
        """
        Erstellt ein wiederverwendbares Voice-Clone-Prompt-Objekt.
        Effizient für mehrfache Generierung mit derselben Stimme.
        """
        model = self.load_clone_model(progress_cb)
        if progress_cb:
            progress_cb("Erstelle Voice-Clone-Prompt…")
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        self._sync_device()
        return prompt

    def generate_with_prompt(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str]],
        voice_clone_prompt: object,
        progress_cb: Optional[Callable[[str], None]] = None,
        max_new_tokens: int = 4092,
        temperature: float = 1.8,
        repetition_penalty: float = 1.05,
        subtalker_temperature: float = 1.8,
    ) -> tuple[list[np.ndarray], int]:
        """Generiert Sprache mit einem gespeicherten Voice-Clone-Prompt."""
        model = self.load_clone_model(progress_cb)
        if progress_cb:
            progress_cb("Generiere mit gespeichertem Stimmenprofil…")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=1.6, #temperature,
            repetition_penalty=1.1, #repetition_penalty,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=1.6, #subtalker_temperature,
        )
        self._sync_device()
        return wavs, sr

    # ── Hilfsmethoden ─────────────────────────────────────────────────────
    def _sync_device(self) -> None:
        if self._device == "mps":
            torch.mps.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()
