"""
core/engine.py – TTS-Engine (Singleton)

Verwaltet das Qwen3-TTS-Modell für Inferenz:
  • load_base_model()       → Base-Modell laden (für Training)
  • load_finetuned_model()  → Finetuned CustomVoice-Checkpoint laden
  • generate_custom_voice() → Sprache generieren mit trainierter Stimme

Apple-Silicon / MPS-Hinweise
-----------------------------
• Modell MUSS torch.float32 verwenden (float16 → NaN-Fehler auf MPS).
• attn_implementation="sdpa" statt "flash_attention_2" (kein CUDA auf Mac).
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
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

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
    """Thread-sicherer Singleton für TTS-Inferenz.

    Unterstützt Model-Switching: Es kann immer nur ein Modell gleichzeitig
    geladen sein (Speicherbeschränkung auf 24 GB M4).
    """

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
        self._model = None
        self._model_lock = Lock()
        self._device = _detect_device()
        self._current_checkpoint: Optional[str] = None
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
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def current_checkpoint(self) -> Optional[str]:
        return self._current_checkpoint

    # ── Base-Modell laden (für Training) ──────────────────────────────────
    def load_base_model(
        self, progress_cb: Optional[Callable[[str], None]] = None
    ) -> object:
        """Lädt das Base-Modell (Qwen3-TTS-12Hz-1.7B-Base). Für Training-Zwecke."""
        with self._model_lock:
            if self._model is not None and self._current_checkpoint == BASE_MODEL_ID:
                return self._model
            self.unload_model()
            if progress_cb:
                progress_cb("Lade Base-Modell (1.7B)…")

            from qwen_tts import Qwen3TTSModel  # type: ignore[import]

            self._model = Qwen3TTSModel.from_pretrained(
                BASE_MODEL_ID,
                device_map={"": self._device},
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            self._current_checkpoint = BASE_MODEL_ID
            if progress_cb:
                progress_cb("Base-Modell bereit.")
        return self._model

    # ── Finetuned-Modell laden ────────────────────────────────────────────
    def load_finetuned_model(
        self,
        checkpoint_path: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> object:
        """Lädt einen Finetuned CustomVoice-Checkpoint.

        Entlädt zuerst das aktuelle Modell, da zwei Modelle nicht
        gleichzeitig in 24 GB Speicher passen.
        """
        with self._model_lock:
            if self._current_checkpoint == checkpoint_path and self._model is not None:
                return self._model
            self.unload_model()
            if progress_cb:
                progress_cb("Lade finetuned Modell…")

            from qwen_tts import Qwen3TTSModel  # type: ignore[import]

            self._model = Qwen3TTSModel.from_pretrained(
                checkpoint_path,
                device_map={"": self._device},
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            self._current_checkpoint = checkpoint_path
            if progress_cb:
                progress_cb("Finetuned Modell bereit.")
        return self._model

    # ── Modell entladen ───────────────────────────────────────────────────
    def unload_model(self) -> None:
        """Modell entladen und Speicher freigeben."""
        if self._model is not None:
            del self._model
            self._model = None
            self._current_checkpoint = None
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()

    # ── Inferenz: CustomVoice ─────────────────────────────────────────────
    def generate_custom_voice(
        self,
        text: Union[str, list[str]],
        speaker_name: str,
        language: Union[str, list[str]] = "Auto",
        instruct: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        max_new_tokens: int = 4092,
        temperature: float = 1.8,
        repetition_penalty: float = 1.05,
        subtalker_temperature: float = 1.8,
    ) -> tuple[list[np.ndarray], int]:
        """Generiert Sprache mit einem Finetuned CustomVoice-Modell.

        Args:
            text: Text(e) zum Vorlesen.
            speaker_name: Speaker-Name (wie im Training definiert).
            language: Sprache (Auto, German, English, ...).
            instruct: Optionale Emotions-/Stil-Anweisung (nur 1.7B).
            progress_cb: Fortschritts-Callback.
            max_new_tokens: Maximale Token-Anzahl.
            temperature: Sampling-Temperatur.
            repetition_penalty: Wiederholungs-Penalty.
            subtalker_temperature: Sub-Talker-Temperatur.
        """
        if self._model is None:
            raise RuntimeError("Kein Modell geladen. Lade zuerst einen Checkpoint.")

        if progress_cb:
            progress_cb("Generiere Sprache mit CustomVoice…")

        wavs, sr = self._model.generate_custom_voice(
            text=text,
            speaker=speaker_name,
            language=language,
            instruct=instruct,
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
