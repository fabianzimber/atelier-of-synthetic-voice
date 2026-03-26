"""Modern Flet UI focused on clone-only speech generation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Optional

import flet as ft
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from core.engine import SUPPORTED_LANGUAGES, TTSEngine
from core.profiles import VoiceProfile, VoiceProfileManager

# Patch torchaudio BEFORE importing miner which imports speechbrain
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
import speechbrain.utils.torch_audio_backend

speechbrain.utils.torch_audio_backend.check_torchaudio_backend = lambda: None

from core.extractor import AudioExtractor

APP_TITLE = "Atelier of Synthetic Voice"
TARGET_SR = 22050
SILENCE_GAP_SECONDS = 0.39
WHISPER_MODELS = ["large-v3", "medium", "small", "base", "tiny"]
MEDIA_EXTENSIONS = [
    "wav",
    "mp3",
    "flac",
    "m4a",
    "ogg",
    "aif",
    "aiff",
    "mp4",
    "mov",
    "mkv",
    "avi",
    "webm",
]


@dataclass
class StudioSettings:
    whisper_model: str = "large-v3"
    auto_open_exports: bool = False
    gemini_api_key: str = ""
    max_token_size: int = 2048
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9


def _settings_path() -> Path:
    return Path(__file__).resolve().parent.parent / "voices" / "app_settings.json"


def _load_settings() -> StudioSettings:
    path = _settings_path()
    if not path.exists():
        return StudioSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return StudioSettings(
            whisper_model=data.get("whisper_model", "large-v3"),
            auto_open_exports=bool(data.get("auto_open_exports", False)),
            gemini_api_key=data.get("gemini_api_key", ""),
            max_token_size=int(data.get("max_token_size", 2048)),
            temperature=float(data.get("temperature", 0.9)),
            repetition_penalty=float(data.get("repetition_penalty", 1.05)),
            subtalker_temperature=float(data.get("subtalker_temperature", 0.9)),
        )
    except Exception:
        return StudioSettings()


def _save_settings(settings: StudioSettings) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    common = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // common, src_sr // common).astype(np.float32)


def _load_audio_universal(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return _to_mono(audio), int(sr)
    except Exception:
        pass

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(TARGET_SR),
                "-ac",
                "1",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode(errors="replace")
            raise RuntimeError(error.strip() or "ffmpeg conversion failed")

        audio, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
        return _to_mono(audio), int(sr)
    finally:
        tmp_path.unlink(missing_ok=True)


def _merge_reference_files(paths: list[Path]) -> tuple[np.ndarray, int, float]:
    if not paths:
        raise ValueError("No files selected")

    merged_parts: list[np.ndarray] = []
    silence = np.zeros(int(TARGET_SR * SILENCE_GAP_SECONDS), dtype=np.float32)

    for index, path in enumerate(paths):
        audio, sr = _load_audio_universal(path)
        audio = _resample_audio(audio, sr, TARGET_SR)
        merged_parts.append(audio)
        if index < len(paths) - 1:
            merged_parts.append(silence)

    merged = np.concatenate(merged_parts).astype(np.float32)
    duration = float(len(merged) / TARGET_SR)
    return merged, TARGET_SR, duration


def _transcribe_audio(audio: np.ndarray, sr: int, model_name: str) -> str:
    import whisper  # type: ignore[import]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        sf.write(str(tmp_path), audio, sr)
        model = whisper.load_model(model_name, device="cpu")
        result = model.transcribe(
            str(tmp_path), fp16=False, language=None, verbose=False
        )
        text = result.get("text", "")
        if isinstance(text, str):
            return text.strip()
        return str(text).strip()
    finally:
        tmp_path.unlink(missing_ok=True)


class VoiceCloneStudioApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.engine = TTSEngine()
        self.manager = VoiceProfileManager()
        self.settings = _load_settings()

        self.section_index = 0
        self.reference_files: list[Path] = []
        self.reference_audio: Optional[np.ndarray] = None
        self.reference_sr: int = TARGET_SR
        self.reference_duration: float = 0.0

        self.profiles: list[VoiceProfile] = []
        self.selected_profile_id: Optional[str] = None

        self.last_output: Optional[tuple[np.ndarray, int, Path]] = None
        self.playback_process: Optional[subprocess.Popen] = None

        self.nav_buttons: list[tuple[ft.Container, ft.Icon, ft.Text]] = []

        self.reference_picker = ft.FilePicker()
        self.miner_target_picker = ft.FilePicker()
        self.miner_source_picker = ft.FilePicker()
        self.page.services.extend(
            [self.reference_picker, self.miner_target_picker, self.miner_source_picker]
        )

        self._configure_page()
        self._build_ui()
        self._reload_profiles()
        self._refresh_model_status()

    def _configure_page(self) -> None:
        self.page.title = APP_TITLE
        self.page.bgcolor = "#F8FAFC"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.theme = ft.Theme(
            color_scheme_seed="#007AFF",
            use_material3=True,
            font_family="SF Pro Text",
            visual_density=ft.VisualDensity.COMFORTABLE,
        )
        self.page.on_disconnect = self._on_disconnect
        # Improved responsiveness
        self.page.padding = ft.padding.all(20)
        self.page.window.min_width = 800
        self.page.window.min_height = 600
        self.page.window.icon = "logo.png"

    def _build_ui(self) -> None:
        title = ft.Column(
            spacing=2,
            controls=[
                ft.Text(
                    "Atelier of Synthetic Voice",
                    size=32,
                    weight=ft.FontWeight.W_700,
                    color="#1E293B",
                ),
                ft.Text(
                    "High-Fidelity Voice Cloning Studio",
                    size=13,
                    color="#64748B",
                    weight=ft.FontWeight.W_500,
                ),
            ],
        )

        nav = self._build_navigation()

        self.top_status_text = ft.Text(
            "Engine: Ready", size=11, color="#475569", weight=ft.FontWeight.W_600
        )
        self.top_status = ft.Container(
            padding=ft.padding.symmetric(horizontal=16, vertical=16),
            border_radius=20,
            bgcolor="#F1F5F9",
            border=ft.border.all(1, "#E2E8F0"),
            content=self.top_status_text,
        )

        top_bar = ft.Container(
            margin=ft.margin.only(bottom=24),
            padding=ft.padding.symmetric(horizontal=30, vertical=20),
            border_radius=24,
            bgcolor="#CCFFFFFF",
            blur=ft.Blur(30, 30, ft.BlurTileMode.CLAMP),
            border=ft.border.all(1, "#E2E8F0"),
            shadow=ft.BoxShadow(
                spread_radius=-5,
                blur_radius=20,
                color="#1A000000",
                offset=ft.Offset(0, 10),
            ),
            content=ft.Column(
                spacing=20,
                controls=[
                    ft.Row(
                        [title, self.top_status],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                    nav,
                ],
            ),
        )

        self.content_host = ft.Column(
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=20,
        )

        self.miner_view = self._build_miner_view()
        self.voice_lab_view = self._build_voice_lab()
        self.speech_studio_view = self._build_speech_studio()
        self.system_view = self._build_system_view()
        self._set_section(0, update=False)

        root = ft.Stack(
            expand=True,
            controls=[
                ft.Container(
                    expand=True,
                    gradient=ft.LinearGradient(
                        begin=ft.Alignment.TOP_LEFT,
                        end=ft.Alignment.BOTTOM_RIGHT,
                        colors=["#F8FAFC", "#F1F5F9"],
                    ),
                ),
                ft.SafeArea(
                    expand=True,
                    content=ft.Column(
                        expand=True,
                        spacing=0,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                        controls=[top_bar, self.content_host],
                    ),
                ),
            ],
        )

        self.page.add(root)

    def _build_navigation(self) -> ft.Control:
        nav_host = ft.Row(spacing=10, wrap=True)
        nav_items = [
            ("Audio Extractor", ft.Icons.RADAR),
            ("Voice Lab", ft.Icons.MIC),
            ("Speech Studio", ft.Icons.GRAPHIC_EQ),
            ("System Settings", ft.Icons.SETTINGS),
        ]

        for idx, (label, icon_name) in enumerate(nav_items):
            icon = ft.Icon(icon_name, size=16)
            text = ft.Text(label, size=12, weight=ft.FontWeight.W_600)
            chip = ft.Container(
                border_radius=14,
                padding=ft.padding.symmetric(horizontal=14, vertical=9),
                on_click=lambda e, i=idx: self._set_section(i),
                content=ft.Row([icon, text], spacing=8, tight=True),
            )
            nav_host.controls.append(chip)
            self.nav_buttons.append((chip, icon, text))

        self._refresh_nav_style()
        return nav_host

    def _set_section(self, index: int, update: bool = True) -> None:
        self.section_index = index
        views = [
            self.miner_view,
            self.voice_lab_view,
            self.speech_studio_view,
            self.system_view,
        ]
        if 0 <= index < len(views):
            self.content_host.controls = [views[index]]

        self._refresh_nav_style()
        if update:
            self.page.update()

    def _refresh_nav_style(self) -> None:
        for idx, (chip, icon, text) in enumerate(self.nav_buttons):
            is_active = idx == self.section_index
            chip.bgcolor = "#007AFF" if is_active else "#0D000000"
            icon.color = "#FFFFFF" if is_active else "#64748B"
            text.color = "#FFFFFF" if is_active else "#64748B"

    def _panel(
        self, title: str, subtitle: str, controls: list[ft.Control]
    ) -> ft.Control:
        return ft.Container(
            padding=30,
            border_radius=24,
            bgcolor="#CCFFFFFF",
            blur=ft.Blur(20, 20),
            border=ft.border.all(1, "#E2E8F0"),
            shadow=ft.BoxShadow(
                spread_radius=-5,
                blur_radius=20,
                color="#0D000000",
                offset=ft.Offset(0, 10),
            ),
            content=ft.Column(
                spacing=20,
                controls=[
                    ft.Column(
                        spacing=4,
                        controls=[
                            ft.Text(
                                title,
                                size=24,
                                weight=ft.FontWeight.W_700,
                                color="#1E293B",
                            ),
                            ft.Text(subtitle, size=13, color="#64748B"),
                        ],
                    ),
                    ft.Divider(height=10, color="transparent"),
                    *controls,
                ],
            ),
        )

    def _toast(self, message: str, error: bool = False) -> None:
        snack = ft.SnackBar(
            content=ft.Text(message, color="#FFFFFF"),
            bgcolor="#B42318" if error else "#007AFF",
        )
        self.page.overlay.append(snack)
        snack.open = True
        self.page.update()

    def _set_top_status(self, text: str, status_type: str = "idle") -> None:
        colors = {
            "idle": ("#F1F5F9", "#475569", "#E2E8F0"),
            "ok": ("#ECFDF5", "#059669", "#D1FAE5"),
            "warn": ("#FFFBEB", "#D97706", "#FEF3C7"),
            "error": ("#FEF2F2", "#DC2626", "#FEE2E2"),
        }
        bg, fg, border = colors.get(status_type, colors["idle"])
        self.top_status.bgcolor = bg
        self.top_status.border = ft.border.all(1, border)
        self.top_status_text.value = f"Engine: {text}"
        self.top_status_text.color = fg
        self.page.update()

    def _reload_profiles(self, select_id: Optional[str] = None) -> None:
        self.profiles = self.manager.load_all()
        self.profile_cards.controls.clear()

        options = []
        for p in self.profiles:
            is_selected = p.id == select_id
            if is_selected:
                self.selected_profile_id = p.id

            card = ft.Container(
                padding=16,
                border_radius=16,
                bgcolor="#FFFFFF" if is_selected else "#F8FAFC",
                border=ft.border.all(2, "#007AFF" if is_selected else "#E2E8F0"),
                on_click=lambda e, pid=p.id: self._reload_profiles(select_id=pid),
                content=ft.Row(
                    [
                        ft.Icon(
                            ft.Icons.MIC_EXTERNAL_ON if p.has_preview else ft.Icons.MIC,
                            color="#007AFF" if is_selected else "#94A3B8",
                        ),
                        ft.Column(
                            [
                                ft.Text(
                                    p.name,
                                    weight=ft.FontWeight.W_600,
                                    size=14,
                                    color="#1E293B",
                                ),
                                ft.Text(
                                    f"{p.language} • {p.created_display}",
                                    size=11,
                                    color="#64748B",
                                ),
                            ],
                            spacing=2,
                            expand=True,
                        ),
                    ],
                    spacing=12,
                ),
            )
            self.profile_cards.controls.append(card)
            options.append(ft.dropdown.Option(key=p.id, text=p.name))

        self.speech_profile_dropdown.options = options
        if select_id:
            self.speech_profile_dropdown.value = select_id
            self.selected_profile_label.value = f"Selected: {next((p.name for p in self.profiles if p.id == select_id), 'Unknown')}"

        self.page.update()

    def _build_miner_view(self) -> ft.Control:
        self.miner_target_status = ft.Text(
            "No target voice (5s) selected.", size=12, color="#64748B"
        )
        self.miner_source_status = ft.Text(
            "No source media selected.", size=12, color="#64748B"
        )
        self.miner_progress = ft.ProgressRing(
            width=18, height=18, stroke_width=2, visible=False
        )
        self.miner_status_label = ft.Text("Ready.", size=12, color="#0A5DCC")
        self.miner_stats_label = ft.Text("", size=11, color="#64748B")
        self.miner_results_list = ft.Column(
            spacing=8, scroll=ft.ScrollMode.AUTO, height=250
        )

        self.miner_target_path: Optional[str] = None
        self.miner_source_path: Optional[str] = None
        self.active_miner: Optional[AudioExtractor] = None
        self.miner_extracted_clips: list = []

        return self._panel(
            "Audio Extractor",
            "Extract perfect emotional voice clips from long, noisy videos.",
            [
                ft.Container(
                    padding=16,
                    border_radius=16,
                    bgcolor="#CCFFFFFF",
                    border=ft.border.all(1, "#B3FFFFFF"),
                    content=ft.Column(
                        spacing=12,
                        controls=[
                            ft.Row(
                                [
                                    ft.ElevatedButton(
                                        "1. Target Voice (5s)",
                                        icon=ft.Icons.PERSON,
                                        on_click=self._pick_miner_target,
                                    ),
                                    self.miner_target_status,
                                ]
                            ),
                            ft.Row(
                                [
                                    ft.ElevatedButton(
                                        "2. Source Media (Long)",
                                        icon=ft.Icons.MOVIE,
                                        on_click=self._pick_miner_source,
                                    ),
                                    self.miner_source_status,
                                ]
                            ),
                            ft.Divider(color="#E5E7EB"),
                            ft.Row(
                                [
                                    ft.ElevatedButton(
                                        "Start Extraction",
                                        icon=ft.Icons.PLAY_ARROW,
                                        on_click=self._start_mining,
                                        style=ft.ButtonStyle(
                                            bgcolor="#007AFF",
                                            color="#FFFFFF",
                                            shape=ft.RoundedRectangleBorder(radius=16),
                                        ),
                                    ),
                                    ft.OutlinedButton(
                                        "Cancel",
                                        icon=ft.Icons.STOP,
                                        on_click=self._cancel_mining,
                                    ),
                                    self.miner_progress,
                                    self.miner_status_label,
                                ]
                            ),
                            self.miner_stats_label,
                        ],
                    ),
                ),
                ft.Text("Extracted Clips", size=15, weight=ft.FontWeight.W_600),
                ft.Container(
                    padding=16,
                    border_radius=16,
                    bgcolor="#F5F8FF",
                    content=self.miner_results_list,
                ),
            ],
        )

    async def _pick_miner_target(self, _: ft.ControlEvent) -> None:
        files = await self.miner_target_picker.pick_files(
            allow_multiple=False, allowed_extensions=MEDIA_EXTENSIONS
        )
        if files and files[0].path:
            self.miner_target_path = files[0].path
            self.miner_target_status.value = (
                f"Selected: {Path(self.miner_target_path).name}"
            )
            self.page.update()

    async def _pick_miner_source(self, _: ft.ControlEvent) -> None:
        files = await self.miner_source_picker.pick_files(
            allow_multiple=False, allowed_extensions=MEDIA_EXTENSIONS
        )
        if files and files[0].path:
            self.miner_source_path = files[0].path
            self.miner_source_status.value = (
                f"Selected: {Path(self.miner_source_path).name}"
            )
            self.page.update()

    def _cancel_mining(self, _: ft.ControlEvent) -> None:
        if self.active_miner:
            self.active_miner.cancel()
            self.miner_status_label.value = "Cancelling..."
            self.page.update()

    async def _start_mining(self, _: ft.ControlEvent) -> None:
        if not self.miner_target_path or not self.miner_source_path:
            self._toast("Please select both target voice and source media.", error=True)
            return
        self.miner_progress.visible = True
        self.miner_status_label.value = "Initializing Extractor..."
        self.miner_results_list.controls.clear()
        self.miner_extracted_clips.clear()
        self.page.update()

        output_dir = Path(__file__).resolve().parent.parent / "extracted_clips"
        self.active_miner = AudioExtractor(
            target_audio_path=self.miner_target_path,
            source_media_path=self.miner_source_path,
            output_dir=str(output_dir),
            gemini_api_key=self.settings.gemini_api_key or None,
        )

        def status_cb(msg: str):
            self.miner_status_label.value = msg
            self.page.update()

        def stats_cb(stats: dict):
            s = f"Chunks Processed: {stats['processed']}/{stats['total']} | Kept: {stats['found']}"
            self.miner_stats_label.value = s
            self.page.update()

        try:
            async for clip in self.active_miner.extract(status_cb, stats_cb):
                self.miner_extracted_clips.append(clip)
                self.miner_results_list.controls.append(
                    self._build_clip_row(clip)
                )
                self.page.update()
        except Exception as e:
            self._toast(f"Extraction error: {e}", error=True)
        finally:
            # Post-filter: keep top 3 per emotion, delete the rest from disk
            before = len(self.miner_extracted_clips)
            self.miner_extracted_clips = AudioExtractor.filter_top_per_emotion(
                self.miner_extracted_clips, max_per_emotion=3
            )
            removed = before - len(self.miner_extracted_clips)
            if removed > 0:
                self.miner_results_list.controls.clear()
                for clip in self.miner_extracted_clips:
                    self.miner_results_list.controls.append(
                        self._build_clip_row(clip)
                    )

            self.miner_progress.visible = False
            cancelled = self.active_miner and self.active_miner.is_cancelled
            msg = "Cancelled." if cancelled else "Extraction Complete."
            if removed > 0:
                msg += f" Filtered: kept top 3/emotion, removed {removed} clips."
            self.miner_status_label.value = msg
            self.active_miner = None
            self.page.update()

    def _build_clip_row(self, clip) -> ft.Container:
        dur = clip.duration
        if dur >= 15.0:
            badge_bg, badge_fg, range_label = "#DCFCE7", "#065F46", "Optimal"
        elif dur >= 8.0:
            badge_bg, badge_fg, range_label = "#FEF3C7", "#78350F", "OK"
        else:
            badge_bg, badge_fg, range_label = "#FEE2E2", "#991B1B", "Short"

        similarity_pct = int(clip.speaker_similarity * 100) if clip.speaker_similarity > 0 else 0

        return ft.Container(
            padding=10,
            border_radius=10,
            bgcolor="#F8FAFE",
            border=ft.border.all(1, "#DDE5F0"),
            content=ft.Row(
                [
                    ft.IconButton(
                        ft.Icons.PLAY_CIRCLE_FILL,
                        icon_color="#0A84FF",
                        on_click=lambda e, p=clip.path: asyncio.create_task(
                            self._play_mined_clip(p)
                        ),
                    ),
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        f"{clip.emotion}  ·  {clip.clarity_score}/10",
                                        weight=ft.FontWeight.W_600,
                                        size=12,
                                    ),
                                    ft.Container(
                                        content=ft.Text(
                                            f"{dur:.1f}s · {range_label}",
                                            size=10,
                                            color=badge_fg,
                                        ),
                                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                                        border_radius=6,
                                        bgcolor=badge_bg,
                                    ),
                                    ft.Container(
                                        content=ft.Text(
                                            f"Sim: {similarity_pct}%",
                                            size=10,
                                            color="#475569",
                                        ),
                                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                                        border_radius=6,
                                        bgcolor="#F1F5F9",
                                    ),
                                ],
                                spacing=6,
                            ),
                            # Sub-score detail row (only shown when Gemini scored the clip)
                            ft.Row(
                                [
                                    ft.Text(
                                        f"Q:{clip.audio_quality}  E:{clip.expressiveness}  C:{clip.speech_clarity}",
                                        size=10,
                                        color="#94A3B8",
                                    ),
                                ],
                                visible=clip.audio_quality > 0,
                            ),
                            ft.Text(
                                f'"{clip.transcript}"',
                                size=11,
                                italic=True,
                                max_lines=1,
                            ),
                        ],
                        expand=True,
                    ),
                    ft.ElevatedButton(
                        "Use in Lab",
                        on_click=lambda e, c=clip: asyncio.create_task(
                            self._import_mined_clip(c)
                        ),
                    ),
                ]
            ),
        )

    async def _play_mined_clip(self, path: str) -> None:
        self._stop_playback_process()
        try:
            import sys

            cmd = (
                ["afplay", path]
                if sys.platform == "darwin"
                else ["ffplay", "-nodisp", "-autoexit", path]
            )
            self.playback_process = subprocess.Popen(cmd)
        except Exception as e:
            self._toast(f"Playback failed: {e}", error=True)

    async def _import_mined_clip(self, clip) -> None:
        self.reference_files = [Path(clip.path)]
        self.ref_text_input.value = clip.transcript
        self._set_section(1)
        self.merge_progress.visible = True
        self.ref_status.value = "Importing clip..."
        self.page.update()
        try:
            merged, sr, duration = await asyncio.to_thread(
                _merge_reference_files, [Path(clip.path)]
            )
            self.reference_audio, self.reference_sr, self.reference_duration = (
                merged,
                sr,
                duration,
            )
            self.ref_files_list.controls = [
                ft.Text(f"• {Path(clip.path).name}", size=12)
            ]
            self.ref_status.value = f"Ready: 1 clip, {duration:.1f}s."
            self._toast("Clip imported!")
        except Exception as e:
            self._toast(f"Import failed: {e}", error=True)
        finally:
            self.merge_progress.visible = False
            self.page.update()

    async def _build_optimal_reference_from_clips(self) -> None:
        if not self.miner_extracted_clips:
            self._toast("No extracted clips available. Run the Audio Extractor first.", error=True)
            return

        self.merge_progress.visible = True
        self.ref_status.value = "Building optimal reference..."
        self.page.update()

        try:
            output_dir = Path(__file__).resolve().parent.parent / "extracted_clips"
            result = await asyncio.to_thread(
                AudioExtractor.build_optimal_reference,
                self.miner_extracted_clips,
                output_dir,
            )
            if result is None:
                self._toast("Could not build reference: no suitable clips.", error=True)
                return

            out_path, combined_transcript = result
            self.reference_files = [out_path]
            merged, sr, duration = await asyncio.to_thread(
                _merge_reference_files, [out_path]
            )
            self.reference_audio, self.reference_sr, self.reference_duration = (
                merged, sr, duration,
            )
            self.ref_files_list.controls = [
                ft.Text(f"• {out_path.name}", size=12)
            ]
            self.ref_text_input.value = combined_transcript
            self.ref_status.value = f"Optimal reference ready: {duration:.1f}s."
            self._toast(f"Built optimal reference ({duration:.1f}s).")
        except Exception as e:
            self._toast(f"Build failed: {e}", error=True)
        finally:
            self.merge_progress.visible = False
            self.page.update()

    def _build_voice_lab(self) -> ft.Control:
        self.ref_status = ft.Text(
            "No reference media selected.", size=12, color="#64748B"
        )
        self.ref_files_list = ft.Column(
            spacing=4, scroll=ft.ScrollMode.AUTO, height=120
        )
        self.merge_progress = ft.ProgressRing(
            width=18, height=18, stroke_width=2, visible=False
        )
        self.whisper_dropdown = ft.Dropdown(
            label="Whisper model",
            width=170,
            options=[ft.dropdown.Option(k) for k in WHISPER_MODELS],
            value=self.settings.whisper_model,
            on_select=self._on_whisper_model_changed,
        )
        self.ref_text_input = ft.TextField(
            label="Reference transcript", multiline=True, min_lines=3, border_radius=12
        )
        self.profile_name_input = ft.TextField(label="Profile name", border_radius=12)
        self.profile_lang_dropdown = ft.Dropdown(
            label="Language",
            options=[ft.dropdown.Option(l) for l in SUPPORTED_LANGUAGES],
            value="German",
        )
        self.profile_notes_input = ft.TextField(
            label="Notes", multiline=True, min_lines=2, border_radius=12
        )

        self.selected_profile_label = ft.Text("No profile selected.", size=12)
        self.profile_cards = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO, height=300)
        return self._panel(
            "Voice Lab",
            "Create reusable voice profiles.",
            [
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "Select Files",
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=self._pick_reference_files,
                        ),
                        ft.OutlinedButton(
                            "Clear", icon=ft.Icons.CLEAR, on_click=self._clear_reference
                        ),
                        self.merge_progress,
                    ]
                ),
                ft.ElevatedButton(
                    "Build Optimal Reference",
                    icon=ft.Icons.AUTO_FIX_HIGH,
                    on_click=lambda e: asyncio.create_task(
                        self._build_optimal_reference_from_clips()
                    ),
                    tooltip="Auto-compose the best ~28s reference from extracted clips for maximum emotional variety",
                    style=ft.ButtonStyle(
                        bgcolor="#0A84FF",
                        color="#FFFFFF",
                        shape=ft.RoundedRectangleBorder(radius=12),
                    ),
                ),
                self.ref_status,
                ft.Container(
                    height=120,
                    padding=10,
                    border_radius=12,
                    bgcolor="#FFFFFF",
                    border=ft.border.all(1, "#E2E8F0"),
                    content=self.ref_files_list,
                ),
                ft.Row(
                    [
                        self.whisper_dropdown,
                        ft.ElevatedButton(
                            "Auto Transcribe", on_click=self._transcribe_reference
                        ),
                    ]
                ),
                self.ref_text_input,
                ft.Divider(),
                self.profile_name_input,
                self.profile_lang_dropdown,
                self.profile_notes_input,
                ft.ElevatedButton(
                    "Create Profile",
                    icon=ft.Icons.AUTO_AWESOME,
                    on_click=self._create_profile,
                    style=ft.ButtonStyle(
                        bgcolor="#111827",
                        color="#FFFFFF",
                        shape=ft.RoundedRectangleBorder(radius=12),
                    ),
                ),
                ft.Divider(),
                self.selected_profile_label,
                self.profile_cards,
                ft.Row(
                    [
                        ft.OutlinedButton(
                            "Delete Selected",
                            on_click=self._delete_selected_profile,
                            style=ft.ButtonStyle(color="#B42318"),
                        ),
                        ft.ElevatedButton(
                            "Open in Studio", on_click=self._jump_to_speech
                        ),
                    ]
                ),
            ],
        )

    async def _pick_reference_files(self, _: ft.ControlEvent) -> None:
        files = await self.reference_picker.pick_files(
            allow_multiple=True, allowed_extensions=MEDIA_EXTENSIONS
        )
        if not files:
            return
        self.reference_files = [Path(f.path) for f in files if f.path]
        self.merge_progress.visible = True
        self.ref_status.value = "Merging..."
        self.page.update()
        try:
            merged, sr, duration = await asyncio.to_thread(
                _merge_reference_files, self.reference_files
            )
            self.reference_audio, self.reference_sr, self.reference_duration = (
                merged,
                sr,
                duration,
            )
            self.ref_files_list.controls = [
                ft.Text(f"• {f.name}", size=12) for f in self.reference_files
            ]
            self.ref_status.value = (
                f"Ready: {len(self.reference_files)} files, {duration:.1f}s."
            )
        except Exception as e:
            self._toast(f"Merge failed: {e}", error=True)
        finally:
            self.merge_progress.visible = False
            self.page.update()

    def _clear_reference(self, _: Optional[ft.ControlEvent]) -> None:
        self.reference_files, self.reference_audio = [], None
        self.ref_status.value = "No reference media selected."
        self.ref_files_list.controls.clear()
        self.ref_text_input.value = ""
        self.page.update()

    async def _transcribe_reference(self, _: ft.ControlEvent) -> None:
        if self.reference_audio is None:
            return
        self.merge_progress.visible = True
        self.ref_status.value = "Transcribing..."
        self.page.update()
        try:
            text = await asyncio.to_thread(
                _transcribe_audio,
                self.reference_audio,
                self.reference_sr,
                self.settings.whisper_model,
            )
            self.ref_text_input.value = text
            self.ref_status.value = "Transcription complete."
        except Exception as e:
            self._toast(f"Whisper failed: {e}", error=True)
        finally:
            self.merge_progress.visible = False
            self.page.update()

    async def _create_profile(self, _: ft.ControlEvent) -> None:
        name, text = (
            self.profile_name_input.value.strip(),
            self.ref_text_input.value.strip(),
        )
        if not name or not text or self.reference_audio is None:
            self._toast("Name, text and audio required.", error=True)
            return
        try:
            p = self.manager.create(
                name,
                text,
                self.profile_lang_dropdown.value,
                self.reference_audio,
                self.reference_sr,
                self.profile_notes_input.value,
            )
            self._reload_profiles(p.id)
            self._toast(f"Profile '{name}' created!")
            self._clear_reference(None)
        except Exception as e:
            self._toast(f"Failed: {e}", error=True)

    def _delete_selected_profile(self, _: ft.ControlEvent) -> None:
        if self.selected_profile_id:
            self.manager.delete(self.selected_profile_id)
            self.selected_profile_id = None
            self._reload_profiles()
            self._toast("Profile deleted.")

    def _jump_to_speech(self, _: ft.ControlEvent) -> None:
        if self.selected_profile_id:
            self.speech_profile_dropdown.value = self.selected_profile_id
            self._set_section(2)

    def _build_speech_studio(self) -> ft.Control:
        self.speech_profile_dropdown = ft.Dropdown(label="Profile", border_radius=12)
        self.speech_lang_dropdown = ft.Dropdown(
            label="Language",
            options=[ft.dropdown.Option(l) for l in SUPPORTED_LANGUAGES],
            value="German",
        )
        self.speech_text_input = ft.TextField(
            label="Text to speak", multiline=True, min_lines=5, border_radius=12
        )
        self.generate_status = ft.Text("Ready.", size=12)
        self.generate_progress = ft.ProgressRing(
            width=18, height=18, stroke_width=2, visible=False
        )
        self.output_label = ft.Text("No output.", size=12)

        return self._panel(
            "Speech Studio",
            "Generate high-quality synthetic speech.",
            [
                ft.Row([self.speech_profile_dropdown, self.speech_lang_dropdown]),
                self.speech_text_input,
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "Generate",
                            icon=ft.Icons.BOLT,
                            on_click=self._generate_speech,
                        ),
                        self.generate_progress,
                        self.generate_status,
                    ]
                ),
                ft.Container(
                    padding=15,
                    border_radius=12,
                    bgcolor="#FFFFFF",
                    border=ft.border.all(1, "#E2E8F0"),
                    content=ft.Column(
                        [
                            ft.Text("Output", weight=ft.FontWeight.W_600),
                            self.output_label,
                            ft.Row(
                                [
                                    ft.IconButton(
                                        ft.Icons.PLAY_ARROW, on_click=self._play_output
                                    ),
                                    ft.IconButton(
                                        ft.Icons.STOP, on_click=self._stop_output
                                    ),
                                    ft.IconButton(
                                        ft.Icons.FOLDER_OPEN, on_click=self._open_output
                                    ),
                                    ft.IconButton(
                                        ft.Icons.BOOKMARK_ADD,
                                        on_click=self._save_preview,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ),
            ],
        )

    async def _generate_speech(self, _: ft.ControlEvent) -> None:
        pid, text = (
            self.speech_profile_dropdown.value,
            self.speech_text_input.value.strip(),
        )
        if not pid or not text:
            return
        p = self.manager.get(pid)
        self.generate_progress.visible = True
        self.generate_status.value = "Generating..."
        self.page.update()
        try:
            ref_audio, sr = await asyncio.to_thread(sf.read, str(p.ref_audio_path))
            wavs, out_sr = await asyncio.to_thread(
                self.engine.generate_with_clone,
                text,
                self.speech_lang_dropdown.value,
                (ref_audio, sr),
                p.ref_text,
                max_new_tokens=self.settings.max_token_size,
                temperature=self.settings.temperature,
                repetition_penalty=self.settings.repetition_penalty,
                subtalker_temperature=self.settings.subtalker_temperature,
            )
            combined = np.concatenate(wavs) if isinstance(wavs, list) else wavs
            out_path = (
                Path(__file__).resolve().parent.parent
                / "exports"
                / f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            out_path.parent.mkdir(exist_ok=True)
            await asyncio.to_thread(sf.write, str(out_path), combined, out_sr)
            self.last_output = (combined, out_sr, out_path)
            self.output_label.value = f"Generated: {out_path.name}"
            if self.settings.auto_open_exports:
                self._open_path(out_path)
        except Exception as e:
            self._toast(f"Failed: {e}", error=True)
        finally:
            self.generate_progress.visible, self.generate_status.value = False, "Ready."
            self.page.update()

    def _play_output(self, _: ft.ControlEvent) -> None:
        if self.last_output:
            self._play_mined_clip(str(self.last_output[2]))

    def _stop_output(self, _: ft.ControlEvent) -> None:
        self._stop_playback_process()

    def _stop_playback_process(self) -> None:
        if self.playback_process:
            self.playback_process.terminate()
            try:
                self.playback_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.playback_process.kill()
        self.playback_process = None

    def _open_output(self, _: ft.ControlEvent) -> None:
        if self.last_output:
            self._open_path(self.last_output[2])

    def _open_path(self, path: Path) -> None:
        import sys

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            pass

    def _save_preview(self, _: ft.ControlEvent) -> None:
        if self.last_output and self.speech_profile_dropdown.value:
            p = self.manager.get(self.speech_profile_dropdown.value)
            self.manager.save_preview(p, self.last_output[0], self.last_output[1])
            self._reload_profiles(p.id)
            self._toast("Preview saved.")

    def _build_system_view(self) -> ft.Control:
        self.model_status_label = ft.Text("Checking...", size=12)

        def make_setting_row(label, control):
            return ft.Row(
                [
                    ft.Text(label, size=13, weight=ft.FontWeight.W_500, expand=True),
                    ft.Container(control, width=200),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            )

        self.system_whisper_dropdown = ft.Dropdown(
            options=[ft.dropdown.Option(k) for k in WHISPER_MODELS],
            value=self.settings.whisper_model,
            on_select=self._on_whisper_model_changed,
            text_size=12,
            height=40,
        )

        self.temp_slider = ft.Slider(
            min=0.1,
            max=2.0,
            divisions=19,
            label="{value}",
            value=self.settings.temperature,
            on_change=self._save_numeric_settings,
        )
        self.rep_penalty_slider = ft.Slider(
            min=1.0,
            max=1.5,
            divisions=10,
            label="{value}",
            value=self.settings.repetition_penalty,
            on_change=self._save_numeric_settings,
        )
        self.sub_temp_slider = ft.Slider(
            min=0.1,
            max=2.0,
            divisions=19,
            label="{value}",
            value=self.settings.subtalker_temperature,
            on_change=self._save_numeric_settings,
        )
        self.max_tokens_input = ft.TextField(
            value=str(self.settings.max_token_size),
            on_change=self._save_numeric_settings,
            text_size=12,
            height=40,
            content_padding=10,
        )

        return self._panel(
            "System Settings",
            "Configure engine and AI parameters.",
            [
                ft.Text("Engine Model", size=15, weight=ft.FontWeight.W_600),
                ft.Container(
                    padding=16,
                    border_radius=16,
                    bgcolor="#F8FAFC",
                    border=ft.border.all(1, "#E2E8F0"),
                    content=ft.Row(
                        [
                            ft.ElevatedButton(
                                "Preload Clone Model",
                                on_click=self._preload_clone_model,
                                icon=ft.Icons.DOWNLOAD,
                            ),
                            self.model_status_label,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ),
                ft.Divider(height=20),
                ft.Text("Generation Parameters", size=15, weight=ft.FontWeight.W_600),
                ft.Container(
                    padding=16,
                    border_radius=16,
                    bgcolor="#F8FAFC",
                    border=ft.border.all(1, "#E2E8F0"),
                    content=ft.Column(
                        spacing=10,
                        controls=[
                            make_setting_row(
                                "Whisper Model", self.system_whisper_dropdown
                            ),
                            make_setting_row("Temperature", self.temp_slider),
                            make_setting_row(
                                "Repetition Penalty", self.rep_penalty_slider
                            ),
                            make_setting_row(
                                "Sub-talker Temperature", self.sub_temp_slider
                            ),
                            make_setting_row("Max Tokens", self.max_tokens_input),
                        ],
                    ),
                ),
                ft.Divider(height=20),
                ft.Text("Studio & API", size=15, weight=ft.FontWeight.W_600),
                ft.Container(
                    padding=16,
                    border_radius=16,
                    bgcolor="#F8FAFC",
                    border=ft.border.all(1, "#E2E8F0"),
                    content=ft.Column(
                        spacing=12,
                        controls=[
                            ft.TextField(
                                label="Gemini API Key",
                                password=True,
                                can_reveal_password=True,
                                value=self.settings.gemini_api_key,
                                on_change=self._save_gemini_api_key,
                                border_radius=12,
                            ),
                            ft.Switch(
                                label="Auto-reveal exports in folder",
                                value=self.settings.auto_open_exports,
                                on_change=self._toggle_auto_open,
                            ),
                        ],
                    ),
                ),
            ],
        )

    def _save_numeric_settings(self, _: ft.ControlEvent) -> None:
        try:
            self.settings.temperature = float(self.temp_slider.value)
            self.settings.repetition_penalty = float(self.rep_penalty_slider.value)
            self.settings.subtalker_temperature = float(self.sub_temp_slider.value)
            self.settings.max_token_size = int(self.max_tokens_input.value or 2048)
            _save_settings(self.settings)
        except ValueError:
            pass

    async def _preload_clone_model(self, _: ft.ControlEvent) -> None:
        self._set_top_status("Loading...", "warn")
        try:
            await asyncio.to_thread(self.engine.load_clone_model)
            self._refresh_model_status()
        except Exception as e:
            self._toast(f"Load failed: {e}", error=True)
            self._set_top_status("Error", "error")

    def _refresh_model_status(self) -> None:
        loaded = self.engine.clone_model_loaded
        self.model_status_label.value = f"Clone model: {'Loaded' if loaded else 'Idle'}"
        self._set_top_status("Ready" if loaded else "Idle", "ok" if loaded else "idle")

    def _toggle_auto_open(self, e: ft.ControlEvent) -> None:
        self.settings.auto_open_exports = bool(e.control.value)
        _save_settings(self.settings)

    def _save_gemini_api_key(self, e: ft.ControlEvent) -> None:
        self.settings.gemini_api_key = e.control.value
        _save_settings(self.settings)

    def _on_whisper_model_changed(self, e: ft.ControlEvent) -> None:
        value = e.control.value
        self.settings.whisper_model = value
        _save_settings(self.settings)
        if hasattr(self, "whisper_dropdown") and self.whisper_dropdown:
            self.whisper_dropdown.value = value
        if hasattr(self, "system_whisper_dropdown") and self.system_whisper_dropdown:
            self.system_whisper_dropdown.value = value
        self.page.update()

    def _on_disconnect(self, _: ft.ControlEvent) -> None:
        self._stop_playback_process()


def _main(page: ft.Page) -> None:
    VoiceCloneStudioApp(page)


def _apply_macos_identity() -> None:
    """Set Dock icon and process name after Flet's engine finishes initializing."""
    import sys
    if sys.platform != "darwin":
        return
    import threading
    import time

    icon_path = str(Path(__file__).resolve().parent.parent / "assets" / "logo.png")

    def _set():
        # Flet's Flutter engine needs ~1 s to finish setting its own icon/name
        time.sleep(1.2)
        try:
            from AppKit import NSApp, NSImage, NSProcessInfo  # type: ignore[import]
            NSProcessInfo.processInfo().setProcessName_(APP_TITLE)
            image = NSImage.alloc().initWithContentsOfFile_(icon_path)
            if image:
                NSApp.setApplicationIconImage_(image)
        except Exception:
            pass

    threading.Thread(target=_set, daemon=True).start()


def run() -> None:
    _apply_macos_identity()
    ft.app(target=_main, view=ft.AppView.FLET_APP, assets_dir="assets")
