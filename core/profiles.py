"""
core/profiles.py – Stimmenprofil-Verwaltung (mit Finetuning-Support)

Jedes Profil besteht aus:
  • Metadaten       → voices/<uuid>/profile.json
  • Referenz-Audio  → voices/<uuid>/reference.wav  (24 kHz, für Training)
  • Trainingsdaten  → voices/<uuid>/training_data/
  • Checkpoints     → voices/<uuid>/checkpoints/
  • Vorschau-Audio  → voices/<uuid>/preview.wav  (optional)
  • Exports         → voices/<uuid>/exports/
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

VOICES_DIR = Path(__file__).parent.parent / "voices"
TRAINING_SR = 24000  # Pflicht für mel_spectrogram im Training (assert sr == 24000)


# ── Datenpflaster ─────────────────────────────────────────────────────────────
@dataclass
class VoiceProfile:
    id: str
    name: str
    ref_text: str
    language: str
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    has_preview: bool = False
    # ── Finetuning-Felder ───────────────────────────────────────────────
    training_status: str = "no_data"  # "no_data" | "data_ready" | "training" | "trained"
    model_path: str = ""              # Relativer Pfad zum besten Checkpoint (z.B. "epoch-2")
    speaker_name: str = ""            # Für generate_custom_voice()
    training_config: dict = field(default_factory=lambda: {
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    })
    training_log: list = field(default_factory=list)  # [{epoch, loss}, ...]
    clip_count: int = 0
    total_duration: float = 0.0

    # ── Pfade ──────────────────────────────────────────────────────────────
    @property
    def directory(self) -> Path:
        return VOICES_DIR / self.id

    @property
    def ref_audio_path(self) -> Path:
        return self.directory / "reference.wav"

    @property
    def preview_audio_path(self) -> Path:
        return self.directory / "preview.wav"

    @property
    def meta_path(self) -> Path:
        return self.directory / "profile.json"

    @property
    def training_data_dir(self) -> Path:
        return self.directory / "training_data"

    @property
    def clips_dir(self) -> Path:
        return self.training_data_dir / "clips"

    @property
    def train_raw_jsonl_path(self) -> Path:
        return self.training_data_dir / "train_raw.jsonl"

    @property
    def train_jsonl_path(self) -> Path:
        return self.training_data_dir / "train.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.directory / "checkpoints"

    @property
    def exports_dir(self) -> Path:
        return self.directory / "exports"

    # ── Serialisierung ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceProfile":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    # ── Anzeige-Hilfsmethoden ──────────────────────────────────────────────
    @property
    def created_display(self) -> str:
        try:
            dt = datetime.fromisoformat(self.created_at)
            return dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            return self.created_at

    @property
    def updated_display(self) -> str:
        try:
            dt = datetime.fromisoformat(self.updated_at)
            return dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            return self.updated_at


# ── Manager ──────────────────────────────────────────────────────────────────
class VoiceProfileManager:
    """Dateisystem-basierte CRUD-Verwaltung für Stimmenprofile mit Finetuning."""

    def __init__(self, voices_dir: Optional[Path] = None) -> None:
        self._root = voices_dir or VOICES_DIR
        self._root.mkdir(parents=True, exist_ok=True)

    # ── Laden ──────────────────────────────────────────────────────────────
    def load_all(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        for meta in sorted(self._root.glob("*/profile.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                profiles.append(VoiceProfile.from_dict(data))
            except Exception:
                continue
        return profiles

    def get(self, profile_id: str) -> Optional[VoiceProfile]:
        meta = self._root / profile_id / "profile.json"
        if not meta.exists():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return VoiceProfile.from_dict(data)
        except Exception:
            return None

    # ── Erstellen ──────────────────────────────────────────────────────────
    def create(
        self,
        name: str,
        language: str,
        ref_audio_source: Path | np.ndarray,
        sample_rate: int = 22050,
        ref_text: str = "",
        notes: str = "",
    ) -> VoiceProfile:
        """Erstellt ein neues Profil. Referenz-Audio wird auf 24 kHz resampled."""
        speaker_name = name.lower().replace(" ", "_").replace("-", "_")
        profile = VoiceProfile(
            id=str(uuid.uuid4()),
            name=name,
            ref_text=ref_text,
            language=language,
            notes=notes,
            speaker_name=speaker_name,
        )
        profile.directory.mkdir(parents=True, exist_ok=True)
        profile.training_data_dir.mkdir(parents=True, exist_ok=True)
        profile.clips_dir.mkdir(parents=True, exist_ok=True)
        profile.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        profile.exports_dir.mkdir(parents=True, exist_ok=True)
        self._save_ref_audio(profile, ref_audio_source, sample_rate)
        self._write_meta(profile)
        return profile

    # ── Aktualisieren ──────────────────────────────────────────────────────
    def update(
        self,
        profile: VoiceProfile,
        name: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
        notes: Optional[str] = None,
        ref_audio_source: Optional[Path | np.ndarray] = None,
        sample_rate: int = 22050,
    ) -> VoiceProfile:
        if name is not None:
            profile.name = name
            profile.speaker_name = name.lower().replace(" ", "_").replace("-", "_")
        if ref_text is not None:
            profile.ref_text = ref_text
        if language is not None:
            profile.language = language
        if notes is not None:
            profile.notes = notes
        if ref_audio_source is not None:
            self._save_ref_audio(profile, ref_audio_source, sample_rate)
            if profile.preview_audio_path.exists():
                profile.preview_audio_path.unlink()
            profile.has_preview = False
        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        self._write_meta(profile)
        return profile

    def save_preview(self, profile: VoiceProfile, audio: np.ndarray, sample_rate: int) -> None:
        sf.write(str(profile.preview_audio_path), audio, sample_rate)
        profile.has_preview = True
        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        self._write_meta(profile)

    # ── Löschen ────────────────────────────────────────────────────────────
    def delete(self, profile_id: str) -> bool:
        directory = self._root / profile_id
        if directory.exists():
            shutil.rmtree(directory)
            return True
        return False

    # ── Trainingsdaten ────────────────────────────────────────────────────
    def add_training_clip(
        self,
        profile: VoiceProfile,
        clip_path: str | Path,
        transcript: str,
        emotion: str = "unknown",
        duration: float = 0.0,
    ) -> None:
        """Kopiert einen Audio-Clip in die Trainingsdaten und ergänzt die raw JSONL."""
        profile.clips_dir.mkdir(parents=True, exist_ok=True)
        src = Path(clip_path)
        dst = profile.clips_dir / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

        entry = {
            "audio": str(dst),
            "text": transcript,
            "ref_audio": str(profile.ref_audio_path),
            "emotion": emotion,
            "duration": duration,
        }
        with open(profile.train_raw_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        profile.clip_count += 1
        profile.total_duration += duration
        if profile.training_status == "no_data":
            profile.training_status = "no_data"  # bleibt no_data bis tokenisiert
        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        self._write_meta(profile)

    def remove_training_clip(self, profile: VoiceProfile, clip_filename: str) -> None:
        """Entfernt einen Clip aus den Trainingsdaten."""
        clip_path = profile.clips_dir / clip_filename
        if clip_path.exists():
            clip_path.unlink()

        # JSONL bereinigen
        if profile.train_raw_jsonl_path.exists():
            lines = profile.train_raw_jsonl_path.read_text(encoding="utf-8").splitlines()
            kept = []
            removed_duration = 0.0
            for line in lines:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if Path(entry["audio"]).name == clip_filename:
                    removed_duration = entry.get("duration", 0.0)
                    continue
                kept.append(line)
            profile.train_raw_jsonl_path.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")
            profile.clip_count = max(0, profile.clip_count - 1)
            profile.total_duration = max(0.0, profile.total_duration - removed_duration)

        # Auch tokenisierte JSONL bereinigen
        if profile.train_jsonl_path.exists():
            lines = profile.train_jsonl_path.read_text(encoding="utf-8").splitlines()
            kept = [l for l in lines if l.strip() and Path(json.loads(l)["audio"]).name != clip_filename]
            profile.train_jsonl_path.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")

        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        self._write_meta(profile)

    def get_training_clips(self, profile: VoiceProfile) -> list[dict]:
        """Liest alle Trainingsclip-Einträge aus der raw JSONL."""
        if not profile.train_raw_jsonl_path.exists():
            return []
        clips = []
        for line in profile.train_raw_jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                clips.append(json.loads(line))
        return clips

    def update_training_status(self, profile: VoiceProfile, status: str, **kwargs) -> None:
        """Aktualisiert den Training-Status und optionale Felder."""
        profile.training_status = status
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        self._write_meta(profile)

    def get_checkpoint_dirs(self, profile: VoiceProfile) -> list[Path]:
        """Liste aller Checkpoint-Verzeichnisse, sortiert nach Epoch."""
        if not profile.checkpoints_dir.exists():
            return []
        dirs = [d for d in profile.checkpoints_dir.iterdir() if d.is_dir()]
        dirs.sort(key=lambda d: d.name)
        return dirs

    def delete_checkpoint(self, profile: VoiceProfile, checkpoint_name: str) -> bool:
        """Entfernt ein bestimmtes Checkpoint-Verzeichnis."""
        cp_dir = profile.checkpoints_dir / checkpoint_name
        if cp_dir.exists():
            shutil.rmtree(cp_dir)
            return True
        return False

    # ── Hilfsmethoden ─────────────────────────────────────────────────────
    def _write_meta(self, profile: VoiceProfile) -> None:
        profile.meta_path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_ref_audio(
        self,
        profile: VoiceProfile,
        source: Path | np.ndarray,
        sample_rate: int = 22050,
    ) -> None:
        """Speichert Referenz-Audio resampled auf 24 kHz (Pflicht für Training)."""
        import librosa

        if isinstance(source, np.ndarray):
            audio = source.astype(np.float32)
            sr = sample_rate
        else:
            audio, sr = sf.read(str(source))
            audio = audio.astype(np.float32)

        # Mono erzwingen
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        # Auf 24 kHz resampled (Pflicht für extract_mels)
        if sr != TRAINING_SR:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=TRAINING_SR)

        sf.write(str(profile.ref_audio_path), audio, TRAINING_SR)

    def name_exists(self, name: str, exclude_id: Optional[str] = None) -> bool:
        for p in self.load_all():
            if p.name == name and p.id != exclude_id:
                return True
        return False
