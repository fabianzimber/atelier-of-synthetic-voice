"""
core/dataset.py – Training-Dataset für Qwen3-TTS Finetuning

Adaptiert von third_party/Qwen3-TTS/finetuning/dataset.py mit:
  • MPS-kompatibles float32
  • Integriertes Ref-Audio Resampling auf 24 kHz

Die collate_fn baut die Dual-Channel-Struktur:
  - Channel 0: Text-Token-IDs
  - Channel 1: Codec-Token-IDs (audio_codes[:,0])
  - Plus: codec_ids (alle 16 Codebooks), ref_mels, attention_mask, labels
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
]

MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    """Training-Dataset für Qwen3-TTS SFT (Supervised Fine-Tuning)."""

    def __init__(self, data_list: list[dict], processor, config: Qwen3TTSConfig):
        self.data_list = data_list
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(
        self, audios: Union[AudioLike, List[AudioLike]]
    ) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]
        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = inputs["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        assert sr == 24000, f"Only support 24kHz audio, got {sr}"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0).float(),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx: int) -> dict:
        item = self.data_list[idx]

        text = item["text"]
        audio_codes = item["audio_codes"]
        ref_audio_path = item["ref_audio"]

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:, :-5],     # (1, text_len)
            "audio_codes": audio_codes,        # (seq_len, 16)
            "ref_mel": ref_mel,                # (1, mel_len, 128)
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        """Baut die Dual-Channel-Input-Struktur für das Training."""
        item_length = [
            b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch
        ]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # Text-Channel (Channel 0)
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = (
                self.config.tts_pad_token_id
            )
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            # Codec-Channel (Channel 1)
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # Placeholder für Speaker-Embedding
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = (
                audio_codec_0
            )
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = (
                self.config.talker_config.codec_eos_token_id
            )

            # Labels (Codec 0 – Next-Token-Prediction)
            codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = (
                audio_codec_0
            )
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = (
                self.config.talker_config.codec_eos_token_id
            )

            # Alle 16 Codebooks
            codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = (
                audio_codecs
            )

            # Masken
            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # Speaker-Embedding-Position
            codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        ref_mels = torch.cat([data["ref_mel"] for data in batch], dim=0)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }
