"""
core/tokenizer.py – Audio-Tokenizer (Singleton)

Wrapper um Qwen3TTSTokenizer (12 Hz) für die Konvertierung von
Audio-Dateien in diskrete audio_codes (Shape: [seq_len, 16]).

Die audio_codes werden als Python-Listen gespeichert (JSON-serialisierbar)
und dienen als Trainings-Input für das Finetuning.
"""

from __future__ import annotations

import warnings
from threading import Lock
from typing import Callable, Optional

import torch

warnings.filterwarnings("ignore", message=".*flash.attn.*")
warnings.filterwarnings("ignore", message=".*FlashAttention.*")

TOKENIZER_MODEL_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class AudioTokenizer:
    """Thread-sicherer Singleton für den Qwen3-TTS Speech-Tokenizer (12 Hz).

    Lazy-loaded: Das Modell wird erst beim ersten ``load()``-Aufruf geladen.
    Nach der Tokenisierung kann ``unload()`` den Speicher freigeben.
    """

    _instance: Optional["AudioTokenizer"] = None
    _creation_lock: Lock = Lock()

    def __new__(cls) -> "AudioTokenizer":
        with cls._creation_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._tokenizer = None
        self._model_lock = Lock()
        self._device = _detect_device()
        self._initialized = True

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None

    def load(self, progress_cb: Optional[Callable[[str], None]] = None):
        """Lade den Speech-Tokenizer. MPS-kompatibel."""
        with self._model_lock:
            if self._tokenizer is not None:
                return self._tokenizer
            if progress_cb:
                progress_cb("Lade Speech-Tokenizer (12 Hz)…")

            from qwen_tts import Qwen3TTSTokenizer  # type: ignore[import]

            self._tokenizer = Qwen3TTSTokenizer.from_pretrained(
                TOKENIZER_MODEL_ID,
                device_map={"": self._device},
            )
            if progress_cb:
                progress_cb("Speech-Tokenizer bereit.")
        return self._tokenizer

    def unload(self) -> None:
        """Tokenizer entladen und Speicher freigeben."""
        with self._model_lock:
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
                if self._device == "mps":
                    torch.mps.empty_cache()
                elif self._device == "cuda":
                    torch.cuda.empty_cache()

    def encode_audio(self, audio_path: str) -> list[list[int]]:
        """Einzelne Audio-Datei → audio_codes als verschachtelte Liste.

        Returns:
            Liste der Form ``[seq_len][16]`` (16 Codebooks bei 12 Hz).
        """
        tokenizer = self.load()
        enc = tokenizer.encode(audio_path, return_dict=True)
        # enc.audio_codes ist eine Liste mit einem Tensor pro Sample
        codes_tensor = enc.audio_codes[0]  # Shape: (seq_len, 16)
        return codes_tensor.cpu().tolist()

    def encode_batch(
        self,
        audio_paths: list[str],
        batch_size: int = 8,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> list[list[list[int]]]:
        """Mehrere Audio-Dateien in Batches tokenisieren.

        Args:
            audio_paths: Liste von WAV-Pfaden.
            batch_size: Batch-Größe (default 8, konservativ für MPS/24 GB).
            progress_cb: Fortschritts-Callback ``f"{done}/{total}"``.

        Returns:
            Liste von audio_codes, je ``[seq_len_i][16]``.
        """
        tokenizer = self.load()
        all_codes: list[list[list[int]]] = []
        total = len(audio_paths)

        for start in range(0, total, batch_size):
            batch = audio_paths[start : start + batch_size]
            enc = tokenizer.encode(batch, return_dict=True)
            for codes_tensor in enc.audio_codes:
                all_codes.append(codes_tensor.cpu().tolist())
            if progress_cb:
                done = min(start + batch_size, total)
                progress_cb(f"Tokenisiert: {done}/{total}")

        return all_codes
