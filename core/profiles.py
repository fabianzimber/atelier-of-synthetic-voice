"""
core/profiles.py – Stimmenprofil-Verwaltung

Jedes Profil besteht aus:
  • Metadaten  → voices/<name>/profile.json
  • Referenz-Audio → voices/<name>/reference.wav
  • Vorschau-Audio  → voices/<name>/preview.wav  (optional, nach erstem Test)

JSON-Schema:
{
  "id":          "uuid4-string",
  "name":        "Meine Stimme",
  "ref_text":    "Transkript des Referenz-Audios",
  "language":    "German",
  "notes":       "Optionale Notizen zur Qualität",
  "created_at":  "2026-02-22T10:00:00",
  "updated_at":  "2026-02-22T10:00:00",
  "has_preview": false
}
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

    # ── Serialisierung ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

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
    """Einfache Dateisystem-basierte CRUD-Verwaltung für Stimmenprofile."""

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
        ref_text: str,
        language: str,
        ref_audio_source: Path | np.ndarray,
        sample_rate: int = 22050,
        notes: str = "",
    ) -> VoiceProfile:
        """Erstellt ein neues Profil und kopiert/speichert das Referenz-Audio."""
        profile = VoiceProfile(
            id=str(uuid.uuid4()),
            name=name,
            ref_text=ref_text,
            language=language,
            notes=notes,
        )
        profile.directory.mkdir(parents=True, exist_ok=True)
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
        """Aktualisiert Metadaten und/oder Referenz-Audio eines Profils."""
        if name is not None:
            profile.name = name
        if ref_text is not None:
            profile.ref_text = ref_text
        if language is not None:
            profile.language = language
        if notes is not None:
            profile.notes = notes
        if ref_audio_source is not None:
            self._save_ref_audio(profile, ref_audio_source, sample_rate)
            # Vorschau löschen, da die Referenz geändert wurde
            if profile.preview_audio_path.exists():
                profile.preview_audio_path.unlink()
            profile.has_preview = False
        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        self._write_meta(profile)
        return profile

    def save_preview(
        self,
        profile: VoiceProfile,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Speichert eine Vorschau-Generierung für ein Profil."""
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
        if isinstance(source, np.ndarray):
            sf.write(str(profile.ref_audio_path), source, sample_rate)
        else:
            # Einlesen und als WAV mit konsistenter Qualität neu schreiben
            audio, sr = sf.read(str(source))
            sf.write(str(profile.ref_audio_path), audio, sr)

    def name_exists(self, name: str, exclude_id: Optional[str] = None) -> bool:
        for p in self.load_all():
            if p.name == name and p.id != exclude_id:
                return True
        return False
