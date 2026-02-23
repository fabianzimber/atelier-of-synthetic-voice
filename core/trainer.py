"""
core/trainer.py – MPS-kompatibler Voice-Trainer

Adaptiert von third_party/Qwen3-TTS/finetuning/sft_12hz.py mit:
  • Kein HuggingFace Accelerator (single-device MPS)
  • torch.float32 (Pflicht auf Apple Silicon)
  • attn_implementation="sdpa" (kein FlashAttention auf Mac)
  • Gradient-Checkpointing für Speichereffizienz
  • Manuelles Gradient-Accumulation
  • Progress-Callbacks für Flet-UI
  • Cancel-Flag für Trainingsabbruch

Speicher-Budget: ~15-18 GB mit Gradient-Checkpointing (passt in 24 GB M4 Pro).
"""

from __future__ import annotations

import json
import os
import shutil

# MPS: Speicher-Limits deaktivieren, sonst Crash bei großen Modellen
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

from .dataset import TTSDataset
from .profiles import VoiceProfile

BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainingConfig:
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    sub_talker_loss_weight: float = 0.3
    gradient_checkpointing: bool = True
    force_cpu: bool = False

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class VoiceTrainer:
    """Single-Device MPS-Trainer für Qwen3-TTS Voice-Finetuning."""

    def __init__(self, profile: VoiceProfile, config: TrainingConfig):
        self.profile = profile
        self.config = config
        self._is_cancelled = False
        self._target_speaker_embedding: Optional[torch.Tensor] = None

    def cancel(self) -> None:
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def train(
        self,
        progress_cb: Optional[Callable[[dict], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        resume_from_epoch: Optional[int] = None,
    ) -> Optional[str]:
        """Trainingsloop ausführen.

        Args:
            progress_cb: Callback mit dict {epoch, step, total_steps, loss, sub_loss}.
            status_cb: Status-Text-Callback.
            resume_from_epoch: Falls gesetzt, wird ab dieser Epoch (exklusive) fortgesetzt.

        Returns:
            Pfad zum letzten Checkpoint oder None bei Abbruch.
        """
        device = "cpu" if self.config.force_cpu else _detect_device()
        dtype = torch.float32  # Pflicht auf MPS

        # ── 1. Modell laden ───────────────────────────────────────────────
        if status_cb:
            status_cb("Lade Modell für Training…")

        from qwen_tts import Qwen3TTSModel  # type: ignore[import]

        # Bestimme Modell-Pfad (Base oder Checkpoint)
        model_id_or_path = BASE_MODEL_ID
        start_epoch = 0

        if resume_from_epoch is not None:
            checkpoint_dir = self.profile.checkpoints_dir / f"epoch-{resume_from_epoch}"
            if checkpoint_dir.exists():
                model_id_or_path = str(checkpoint_dir)
                start_epoch = resume_from_epoch + 1
                if status_cb:
                    status_cb(f"Setze Training ab Epoch {start_epoch} fort…")

        qwen3tts = Qwen3TTSModel.from_pretrained(
            model_id_or_path,
            device_map={"": device},
            dtype=dtype,
            attn_implementation="sdpa",
        )
        model_config = AutoConfig.from_pretrained(BASE_MODEL_ID)
        model = qwen3tts.model

        # Gradient-Checkpointing aktivieren
        if self.config.gradient_checkpointing:
            if hasattr(model.talker, "gradient_checkpointing_enable"):
                model.talker.gradient_checkpointing_enable()

        # ── 2. Dataset laden ──────────────────────────────────────────────
        if status_cb:
            status_cb("Lade Trainingsdaten…")

        train_jsonl = self.profile.train_jsonl_path
        if not train_jsonl.exists():
            raise FileNotFoundError(f"Trainings-JSONL nicht gefunden: {train_jsonl}")

        train_data = []
        for line in train_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                train_data.append(json.loads(line))

        if not train_data:
            raise ValueError("Keine Trainingsdaten in JSONL.")

        dataset = TTSDataset(train_data, qwen3tts.processor, model_config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        # ── 3. Optimizer ──────────────────────────────────────────────────
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # ── 4. Training ──────────────────────────────────────────────────
        model.train()
        training_log = self.profile.training_log or []
        last_checkpoint_path = None

        if status_cb:
            status_cb(
                f"Training gestartet (Epoch {start_epoch + 1}/{self.config.epochs})…"
            )

        for epoch in range(start_epoch, self.config.epochs):
            if self._is_cancelled:
                if status_cb:
                    status_cb("Training abgebrochen.")
                return None

            epoch_losses = []

            for step, batch in enumerate(dataloader):
                if self._is_cancelled:
                    if status_cb:
                        status_cb("Training abgebrochen.")
                    return last_checkpoint_path

                # Batch auf Device verschieben
                input_ids = batch["input_ids"].to(device)
                codec_ids = batch["codec_ids"].to(device)
                ref_mels = batch["ref_mels"].to(device).to(dtype)
                text_embedding_mask = batch["text_embedding_mask"].to(device)
                codec_embedding_mask = batch["codec_embedding_mask"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                codec_0_labels = batch["codec_0_labels"].to(device)
                codec_mask = batch["codec_mask"].to(device)

                # Speaker-Embedding aus Referenz-Mel extrahieren
                # Wir detach'en hier, um Speicher zu sparen
                with torch.no_grad():
                    speaker_embedding = model.speaker_encoder(ref_mels).detach()
                    # Sanity-Check für MPS-Stabilität
                    if torch.isnan(speaker_embedding).any():
                        speaker_embedding = torch.nan_to_num(speaker_embedding, nan=0.0)

                if self._target_speaker_embedding is None:
                    self._target_speaker_embedding = speaker_embedding.clone().cpu()

                # Dual-Channel Embeddings aufbauen
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.model.text_embedding(input_text_ids)
                    * text_embedding_mask
                )
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids)
                    * codec_embedding_mask
                )

                # Speaker-Embedding an Position 6 injizieren
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                # Codec 1-15 Embeddings addieren
                for i in range(1, 16):
                    codec_i_embedding = (
                        model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # Forward: Main Talker
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                # Sub-Talker Loss
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                # Combined Loss
                loss = (
                    outputs.loss + self.config.sub_talker_loss_weight * sub_talker_loss
                )

                # MPS/Stability Check: Skip batch if loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    if status_cb:
                        status_cb(f"Warnung: Batch {step} übersprungen (NaN/Inf Loss)")
                    optimizer.zero_grad()
                    continue

                # Gradient-Accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                # MPS Sync & Cache Clear (Optional but safer)
                if device == "mps" and step % 10 == 0:
                    torch.mps.synchronize()

                loss_val = loss.item()
                sub_loss_val = sub_talker_loss.item()
                epoch_losses.append(loss_val)

                # Progress-Callback – jeden Step
                if progress_cb:
                    progress_cb(
                        {
                            "epoch": epoch,
                            "total_epochs": self.config.epochs,
                            "step": step,
                            "total_steps": len(dataloader),
                            "loss": loss_val,
                            "sub_loss": sub_loss_val,
                        }
                    )

            # Restliche Gradienten anwenden
            optimizer.step()
            optimizer.zero_grad()

            # Epoch-Log
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            training_log.append({"epoch": epoch, "avg_loss": round(avg_loss, 4)})

            if status_cb:
                status_cb(
                    f"Epoch {epoch + 1}/{self.config.epochs} abgeschlossen. Avg Loss: {avg_loss:.4f}"
                )

            # Speicher frei machen für Checkpoint
            if device == "mps":
                torch.mps.empty_cache()

            # Checkpoint speichern
            cp_path = self._save_checkpoint(model, epoch, status_cb=status_cb)
            last_checkpoint_path = cp_path

            # Alte Checkpoints aufräumen (nur letzte 2 behalten)
            self._cleanup_old_checkpoints(keep=2)

        # Training-Log im Profil aktualisieren
        self.profile.training_log = training_log

        if status_cb:
            status_cb("Training abgeschlossen!")

        # Modell freigeben
        del model
        del qwen3tts
        del optimizer
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        return last_checkpoint_path

    def _save_checkpoint(
        self,
        model,
        epoch: int,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Speichert einen vollständigen CustomVoice-Checkpoint.

        Statt das gesamte Base-Modell-Verzeichnis zu kopieren (~3.4 GB, langsam),
        werden nur config.json + model.safetensors geschrieben.
        Alle anderen Dateien (tokenizer etc.) werden per Symlink referenziert.
        """
        if status_cb:
            status_cb(f"Speichere Checkpoint epoch-{epoch}…")

        output_dir = self.profile.checkpoints_dir / f"epoch-{epoch}"
        base_model_path = Path(snapshot_download(BASE_MODEL_ID))

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dateien + Unterverzeichnisse per Symlink referenzieren (spart ~3 GB + Zeit)
        for src_entry in base_model_path.iterdir():
            if src_entry.name in ("config.json", "model.safetensors"):
                continue  # Werden unten überschrieben
            dst = output_dir / src_entry.name
            if src_entry.is_dir():
                # Unterverzeichnisse (z.B. speech_tokenizer/) komplett symlinken
                dst.symlink_to(src_entry.resolve())
            elif src_entry.stat().st_size < 1_000_000:
                shutil.copy2(str(src_entry), str(dst))
            else:
                dst.symlink_to(src_entry.resolve())

        # config.json: base → custom_voice
        base_config = json.loads(
            (base_model_path / "config.json").read_text(encoding="utf-8")
        )
        base_config["tts_model_type"] = "custom_voice"
        talker_config = base_config.get("talker_config", {})
        talker_config["spk_id"] = {self.profile.speaker_name: 3000}
        talker_config["spk_is_dialect"] = {self.profile.speaker_name: False}
        base_config["talker_config"] = talker_config
        (output_dir / "config.json").write_text(
            json.dumps(base_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Gewichte speichern – state_dict() wie in Reference-Implementierung
        if status_cb:
            status_cb(f"Speichere Gewichte für epoch-{epoch}…")

        state_dict = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("speaker_encoder")
        }

        # Speaker-Embedding bei Position 3000 injizieren
        if self._target_speaker_embedding is not None:
            weight = state_dict["talker.model.codec_embedding.weight"]
            weight[3000] = (
                self._target_speaker_embedding[0]
                .detach()
                .cpu()
                .to(weight.dtype)
            )

        save_file(state_dict, str(output_dir / "model.safetensors"))
        del state_dict

        if status_cb:
            status_cb(f"Checkpoint epoch-{epoch} gespeichert.")

        return str(output_dir)

    def _cleanup_old_checkpoints(self, keep: int = 2) -> None:
        """Behält nur die letzten N Checkpoints, löscht ältere."""
        cp_dir = self.profile.checkpoints_dir
        if not cp_dir.exists():
            return
        dirs = sorted(
            [d for d in cp_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if len(dirs) > keep:
            for old_dir in dirs[:-keep]:
                shutil.rmtree(old_dir)
