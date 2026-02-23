"""Flet UI – Finetuning-basierte Voice Synthesis."""

from __future__ import annotations

import asyncio
import json
import subprocess
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
from core.tokenizer import AudioTokenizer
from core.trainer import TrainingConfig, VoiceTrainer

# Patch torchaudio BEFORE importing extractor which imports speechbrain
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
    max_token_size: int = 4092
    temperature: float = 1.8
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 1.8
    force_cpu_training: bool = False


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
            max_token_size=int(data.get("max_token_size", 4092)),
            temperature=float(data.get("temperature", 1.8)),
            repetition_penalty=float(data.get("repetition_penalty", 1.05)),
            subtalker_temperature=float(data.get("subtalker_temperature", 1.8)),
            force_cpu_training=bool(data.get("force_cpu_training", False)),
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

    import tempfile

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
            raise RuntimeError(
                result.stderr.decode(errors="replace").strip() or "ffmpeg failed"
            )
        audio, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
        return _to_mono(audio), int(sr)
    finally:
        tmp_path.unlink(missing_ok=True)


class VoiceSynthStudioApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.engine = TTSEngine()
        self.tokenizer = AudioTokenizer()
        self.manager = VoiceProfileManager()
        self.settings = _load_settings()

        self.section_index = 0
        self.profiles: list[VoiceProfile] = []
        self.selected_profile_id: Optional[str] = None

        self.last_output: Optional[tuple[np.ndarray, int, Path]] = None
        self.playback_process: Optional[subprocess.Popen] = None
        self.active_trainer: Optional[VoiceTrainer] = None

        self.nav_buttons: list[tuple[ft.Container, ft.Icon, ft.Text]] = []

        self.reference_picker = ft.FilePicker()
        self.miner_target_picker = ft.FilePicker()
        self.miner_source_picker = ft.FilePicker()
        self.clip_add_picker = ft.FilePicker()
        self.page.services.extend(
            [
                self.reference_picker,
                self.miner_target_picker,
                self.miner_source_picker,
                self.clip_add_picker,
            ]
        )

        self._configure_page()
        self._build_ui()
        self._reload_profiles()
        self._refresh_model_status()

    # ── Seitenkonfiguration ───────────────────────────────────────────────
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
        self.page.padding = ft.padding.all(20)
        self.page.window.min_width = 800
        self.page.window.min_height = 600
        self.page.window.icon = "logo.png"

    # ── Haupt-UI ─────────────────────────────────────────────────────────
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
                    "Voice Finetuning Studio",
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
            expand=True, scroll=ft.ScrollMode.AUTO, spacing=20
        )

        self.miner_view = self._build_miner_view()
        self.voice_training_view = self._build_voice_training()
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
            ("Voice Training", ft.Icons.MODEL_TRAINING),
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
            self.voice_training_view,
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

    # ── Profil-Verwaltung ────────────────────────────────────────────────
    def _reload_profiles(self, select_id: Optional[str] = None) -> None:
        self.profiles = self.manager.load_all()
        self.profile_cards.controls.clear()

        trained_options = []
        miner_profile_options = [
            ft.dropdown.Option(key="", text="-- Select Profile --")
        ]

        for p in self.profiles:
            is_selected = p.id == select_id
            if is_selected:
                self.selected_profile_id = p.id

            # Status-Icon
            if p.training_status == "trained":
                status_icon = ft.Icons.CHECK_CIRCLE
                status_color = "#059669"
            elif p.training_status in ("training", "data_ready"):
                status_icon = ft.Icons.PENDING
                status_color = "#D97706"
            else:
                status_icon = ft.Icons.RADIO_BUTTON_UNCHECKED
                status_color = "#94A3B8"

            card = ft.Container(
                padding=16,
                border_radius=16,
                bgcolor="#FFFFFF" if is_selected else "#F8FAFC",
                border=ft.border.all(2, "#007AFF" if is_selected else "#E2E8F0"),
                on_click=lambda e, pid=p.id: self._select_profile(pid),
                content=ft.Row(
                    [
                        ft.Icon(status_icon, color=status_color),
                        ft.Column(
                            [
                                ft.Text(
                                    p.name,
                                    weight=ft.FontWeight.W_600,
                                    size=14,
                                    color="#1E293B",
                                ),
                                ft.Text(
                                    f"{p.language} • {p.clip_count} clips • {p.training_status}",
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
            miner_profile_options.append(ft.dropdown.Option(key=p.id, text=p.name))

            if p.training_status == "trained":
                trained_options.append(ft.dropdown.Option(key=p.id, text=p.name))

        # Speech Studio Dropdown – nur trainierte Profile
        self.speech_profile_dropdown.options = trained_options
        if select_id and any(o.key == select_id for o in trained_options):
            self.speech_profile_dropdown.value = select_id

        # Miner Profil-Dropdown
        self.miner_profile_dropdown.options = miner_profile_options

        # Training-Daten aktualisieren wenn Profil ausgewählt
        if self.selected_profile_id:
            self._refresh_training_data_view()

        self.page.update()

    def _select_profile(self, pid: str) -> None:
        self.selected_profile_id = pid
        self._reload_profiles(select_id=pid)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 0: AUDIO EXTRACTOR
    # ══════════════════════════════════════════════════════════════════════
    def _build_miner_view(self) -> ft.Control:
        self.miner_target_status = ft.Text(
            "No target voice selected.", size=12, color="#64748B"
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

        self.miner_profile_dropdown = ft.Dropdown(
            label="Target Profile",
            width=300,
            border_radius=12,
            options=[ft.dropdown.Option(key="", text="-- Select Profile --")],
        )

        return self._panel(
            "Audio Extractor",
            "Extract voice clips from media for finetuning training data.",
            [
                self.miner_profile_dropdown,
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
                ft.Row(
                    [
                        ft.Text("Extracted Clips", size=15, weight=ft.FontWeight.W_600),
                        ft.ElevatedButton(
                            "Add All to Profile",
                            icon=ft.Icons.PLAYLIST_ADD,
                            on_click=self._add_all_clips_to_profile,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
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
            whisper_model_name=self.settings.whisper_model,
        )

        def status_cb(msg: str):
            self.miner_status_label.value = msg
            self.page.update()

        def stats_cb(stats: dict):
            s = f"Processed: {stats['processed']}/{stats['total']} | Found: {stats['found']}"
            self.miner_stats_label.value = s
            self.page.update()

        try:
            async for clip in self.active_miner.extract(status_cb, stats_cb):
                self.miner_extracted_clips.append(clip)
                row = ft.Container(
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
                                    self._play_clip(p)
                                ),
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        f"{clip.emotion} (Score: {clip.clarity_score}/10, "
                                        f"Sim: {clip.similarity_score:.2f})",
                                        weight=ft.FontWeight.W_600,
                                        size=12,
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
                                "Add to Profile",
                                on_click=lambda e, c=clip: asyncio.create_task(
                                    self._add_clip_to_profile(c)
                                ),
                            ),
                        ]
                    ),
                )
                self.miner_results_list.controls.append(row)
                self.page.update()
        except Exception as e:
            self._toast(f"Extraction error: {e}", error=True)
        finally:
            self.miner_progress.visible = False
            self.miner_status_label.value = (
                "Extraction Complete."
                if not self.active_miner.is_cancelled
                else "Cancelled."
            )
            self.active_miner = None
            self.page.update()

    async def _add_clip_to_profile(self, clip) -> None:
        pid = self.miner_profile_dropdown.value
        if not pid:
            self._toast("Please select a target profile first.", error=True)
            return
        profile = self.manager.get(pid)
        if not profile:
            return
        self.manager.add_training_clip(
            profile, clip.path, clip.transcript, clip.emotion, clip.duration
        )
        self._toast(f"Clip added to '{profile.name}'.")
        self._reload_profiles(select_id=self.selected_profile_id)

    async def _add_all_clips_to_profile(self, _: ft.ControlEvent) -> None:
        pid = self.miner_profile_dropdown.value
        if not pid:
            self._toast("Please select a target profile first.", error=True)
            return
        profile = self.manager.get(pid)
        if not profile or not self.miner_extracted_clips:
            self._toast("No clips to add.", error=True)
            return
        count = 0
        for clip in self.miner_extracted_clips:
            self.manager.add_training_clip(
                profile, clip.path, clip.transcript, clip.emotion, clip.duration
            )
            count += 1
        self._toast(f"{count} clips added to '{profile.name}'.")
        self._reload_profiles(select_id=self.selected_profile_id)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: VOICE TRAINING
    # ══════════════════════════════════════════════════════════════════════
    def _build_voice_training(self) -> ft.Control:
        # Profil-Erstellung
        self.train_profile_name = ft.TextField(label="Profile Name", border_radius=12)
        self.train_profile_lang = ft.Dropdown(
            label="Language",
            options=[ft.dropdown.Option(l) for l in SUPPORTED_LANGUAGES],
            value="German",
        )

        self.profile_cards = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO, height=200)

        # Trainingsdaten
        self.training_clips_list = ft.Column(
            spacing=6, scroll=ft.ScrollMode.AUTO, height=200
        )
        self.training_data_stats = ft.Text(
            "No training data.", size=12, color="#64748B"
        )
        self.tokenize_progress = ft.ProgressBar(visible=False, width=400)
        self.tokenize_status = ft.Text("", size=11, color="#64748B")

        # Training-Controls
        self.train_epochs_slider = ft.Slider(
            min=1, max=20, divisions=19, label="{value}", value=3
        )
        self.train_lr_dropdown = ft.Dropdown(
            label="Learning Rate",
            width=200,
            options=[
                ft.dropdown.Option(key="1e-5", text="1e-5 (conservative)"),
                ft.dropdown.Option(key="2e-5", text="2e-5 (recommended)"),
                ft.dropdown.Option(key="5e-5", text="5e-5 (aggressive)"),
            ],
            value="2e-5",
        )
        self.training_progress_text = ft.Text("", size=12, color="#0A5DCC")
        self.training_loss_text = ft.Text("", size=11, color="#64748B")
        self.training_progress_bar = ft.ProgressBar(visible=False, width=400)

        # Checkpoints
        self.checkpoint_list = ft.Column(
            spacing=6, scroll=ft.ScrollMode.AUTO, height=150
        )

        return self._panel(
            "Voice Training",
            "Create profiles, prepare training data, and finetune voice models.",
            [
                # ── Profil erstellen ──
                ft.Text("Create Profile", size=15, weight=ft.FontWeight.W_600),
                ft.Row(
                    [
                        self.train_profile_name,
                        self.train_profile_lang,
                        ft.ElevatedButton(
                            "Select Reference Audio",
                            icon=ft.Icons.AUDIO_FILE,
                            on_click=self._pick_training_reference,
                        ),
                    ]
                ),
                ft.ElevatedButton(
                    "Create Profile",
                    icon=ft.Icons.ADD,
                    on_click=self._create_training_profile,
                    style=ft.ButtonStyle(
                        bgcolor="#111827",
                        color="#FFFFFF",
                        shape=ft.RoundedRectangleBorder(radius=12),
                    ),
                ),
                ft.Divider(),
                # ── Profil-Liste ──
                ft.Text("Profiles", size=15, weight=ft.FontWeight.W_600),
                self.profile_cards,
                ft.Row(
                    [
                        ft.OutlinedButton(
                            "Delete Selected",
                            on_click=self._delete_selected_profile,
                            style=ft.ButtonStyle(color="#B42318"),
                        ),
                    ]
                ),
                ft.Divider(),
                # ── Trainingsdaten ──
                ft.Text("Training Data", size=15, weight=ft.FontWeight.W_600),
                self.training_data_stats,
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "Add Clips Manually",
                            icon=ft.Icons.ADD_CIRCLE,
                            on_click=self._pick_manual_clips,
                        ),
                    ]
                ),
                self.training_clips_list,
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "Prepare Training Data",
                            icon=ft.Icons.MEMORY,
                            on_click=self._prepare_training_data,
                            style=ft.ButtonStyle(
                                bgcolor="#7C3AED",
                                color="#FFFFFF",
                                shape=ft.RoundedRectangleBorder(radius=12),
                            ),
                        ),
                        self.tokenize_status,
                    ]
                ),
                self.tokenize_progress,
                ft.Divider(),
                # ── Training ──
                ft.Text("Training", size=15, weight=ft.FontWeight.W_600),
                ft.Container(
                    padding=16,
                    border_radius=16,
                    bgcolor="#F8FAFC",
                    border=ft.border.all(1, "#E2E8F0"),
                    content=ft.Column(
                        spacing=10,
                        controls=[
                            ft.Row(
                                [
                                    ft.Text(
                                        "Epochs:", size=13, weight=ft.FontWeight.W_500
                                    ),
                                    ft.Container(self.train_epochs_slider, expand=True),
                                ]
                            ),
                            ft.Row([self.train_lr_dropdown]),
                            ft.Row(
                                [
                                    ft.ElevatedButton(
                                        "Start Training",
                                        icon=ft.Icons.PLAY_ARROW,
                                        on_click=self._start_training,
                                        style=ft.ButtonStyle(
                                            bgcolor="#059669",
                                            color="#FFFFFF",
                                            shape=ft.RoundedRectangleBorder(radius=12),
                                        ),
                                    ),
                                    ft.OutlinedButton(
                                        "Stop Training",
                                        icon=ft.Icons.STOP,
                                        on_click=self._stop_training,
                                    ),
                                ]
                            ),
                            self.training_progress_bar,
                            self.training_progress_text,
                            self.training_loss_text,
                        ],
                    ),
                ),
                ft.Divider(),
                # ── Checkpoints ──
                ft.Text("Checkpoints", size=15, weight=ft.FontWeight.W_600),
                self.checkpoint_list,
                ft.ElevatedButton(
                    "Test in Studio",
                    icon=ft.Icons.ARROW_FORWARD,
                    on_click=self._jump_to_studio,
                ),
            ],
        )

    # Referenz-Audio wählen
    self_train_ref_path: Optional[str] = None

    async def _pick_training_reference(self, _: ft.ControlEvent) -> None:
        files = await self.reference_picker.pick_files(
            allow_multiple=False, allowed_extensions=MEDIA_EXTENSIONS
        )
        if files and files[0].path:
            self._train_ref_path = files[0].path
            self._toast(f"Reference: {Path(self._train_ref_path).name}")

    async def _create_training_profile(self, _: ft.ControlEvent) -> None:
        name = self.train_profile_name.value.strip()
        if not name:
            self._toast("Profile name required.", error=True)
            return
        ref_path = getattr(self, "_train_ref_path", None)
        if not ref_path:
            self._toast("Reference audio required.", error=True)
            return
        try:
            audio, sr = _load_audio_universal(Path(ref_path))
            p = self.manager.create(
                name=name,
                language=self.train_profile_lang.value,
                ref_audio_source=audio,
                sample_rate=sr,
            )
            self.train_profile_name.value = ""
            self._train_ref_path = None
            self._reload_profiles(p.id)
            self._toast(f"Profile '{name}' created!")
        except Exception as e:
            self._toast(f"Failed: {e}", error=True)

    def _delete_selected_profile(self, _: ft.ControlEvent) -> None:
        if self.selected_profile_id:
            self.manager.delete(self.selected_profile_id)
            self.selected_profile_id = None
            self._reload_profiles()
            self._toast("Profile deleted.")

    def _refresh_training_data_view(self) -> None:
        """Aktualisiert die Trainingsdaten-Ansicht für das ausgewählte Profil."""
        self.training_clips_list.controls.clear()
        self.checkpoint_list.controls.clear()

        if not self.selected_profile_id:
            self.training_data_stats.value = "No profile selected."
            return

        profile = self.manager.get(self.selected_profile_id)
        if not profile:
            return

        clips = self.manager.get_training_clips(profile)
        total_dur = sum(c.get("duration", 0) for c in clips)
        emotions = {}
        for c in clips:
            em = c.get("emotion", "unknown")
            emotions[em] = emotions.get(em, 0) + 1

        emotion_str = ", ".join(f"{k}({v})" for k, v in sorted(emotions.items()))
        self.training_data_stats.value = (
            f"{len(clips)} clips, {total_dur:.0f}s total"
            f"{f' | {emotion_str}' if emotion_str else ''}"
            f" | Status: {profile.training_status}"
        )

        for clip_data in clips[-50:]:  # Letzte 50 anzeigen
            clip_name = Path(clip_data.get("audio", "")).name
            transcript = clip_data.get("text", "")[:60]
            emotion = clip_data.get("emotion", "?")
            row = ft.Container(
                padding=8,
                border_radius=8,
                bgcolor="#FFFFFF",
                border=ft.border.all(1, "#E2E8F0"),
                content=ft.Row(
                    [
                        ft.IconButton(
                            ft.Icons.PLAY_CIRCLE,
                            icon_size=20,
                            on_click=lambda e, p=clip_data.get("audio", ""): (
                                asyncio.create_task(self._play_clip(p))
                            ),
                        ),
                        ft.Column(
                            [
                                ft.Text(
                                    f"{clip_name} [{emotion}]",
                                    size=11,
                                    weight=ft.FontWeight.W_600,
                                ),
                                ft.Text(transcript, size=10, italic=True, max_lines=1),
                            ],
                            expand=True,
                            spacing=1,
                        ),
                        ft.IconButton(
                            ft.Icons.DELETE_OUTLINE,
                            icon_size=18,
                            icon_color="#B42318",
                            on_click=lambda e, cn=clip_name: asyncio.create_task(
                                self._remove_clip(cn)
                            ),
                        ),
                    ],
                    spacing=4,
                ),
            )
            self.training_clips_list.controls.append(row)

        # Checkpoints
        checkpoints = self.manager.get_checkpoint_dirs(profile)
        for cp in checkpoints:
            self.checkpoint_list.controls.append(
                ft.Container(
                    padding=10,
                    border_radius=10,
                    bgcolor="#F0FDF4",
                    border=ft.border.all(1, "#BBF7D0"),
                    content=ft.Row(
                        [
                            ft.Icon(ft.Icons.SAVE, color="#059669", size=18),
                            ft.Text(
                                cp.name,
                                size=12,
                                weight=ft.FontWeight.W_600,
                                expand=True,
                            ),
                            ft.OutlinedButton(
                                "Load",
                                on_click=lambda e, p=str(cp): asyncio.create_task(
                                    self._load_checkpoint(p)
                                ),
                            ),
                            ft.IconButton(
                                ft.Icons.DELETE_OUTLINE,
                                icon_size=18,
                                on_click=lambda e, n=cp.name: self._delete_checkpoint(
                                    n
                                ),
                            ),
                        ],
                        spacing=8,
                    ),
                )
            )

    async def _remove_clip(self, clip_name: str) -> None:
        if not self.selected_profile_id:
            return
        profile = self.manager.get(self.selected_profile_id)
        if profile:
            self.manager.remove_training_clip(profile, clip_name)
            self._reload_profiles(select_id=self.selected_profile_id)

    async def _pick_manual_clips(self, _: ft.ControlEvent) -> None:
        if not self.selected_profile_id:
            self._toast("Select a profile first.", error=True)
            return
        files = await self.clip_add_picker.pick_files(
            allow_multiple=True, allowed_extensions=["wav", "mp3", "flac", "m4a", "ogg"]
        )
        if not files:
            return
        profile = self.manager.get(self.selected_profile_id)
        if not profile:
            return
        for f in files:
            if f.path:
                self.manager.add_training_clip(
                    profile, f.path, transcript="", emotion="unclassified", duration=0.0
                )
        self._toast(f"{len(files)} clips added.")
        self._reload_profiles(select_id=self.selected_profile_id)

    async def _prepare_training_data(self, _: ft.ControlEvent) -> None:
        if not self.selected_profile_id:
            self._toast("Select a profile first.", error=True)
            return
        profile = self.manager.get(self.selected_profile_id)
        if not profile:
            return

        clips = self.manager.get_training_clips(profile)
        if not clips:
            self._toast("No clips to tokenize.", error=True)
            return

        self.tokenize_progress.visible = True
        self.tokenize_status.value = "Loading tokenizer..."
        self.page.update()

        try:
            # Tokenizer laden
            await asyncio.to_thread(
                self.tokenizer.load, lambda msg: self._update_tokenize_status(msg)
            )

            # Audio-Pfade sammeln
            audio_paths = [c["audio"] for c in clips]
            total = len(audio_paths)

            def progress(msg: str):
                self.tokenize_status.value = msg
                self.tokenize_progress.value = None  # Indeterminate
                self.page.update()

            # Batch-Tokenisierung
            all_codes = await asyncio.to_thread(
                self.tokenizer.encode_batch, audio_paths, 8, progress
            )

            # train.jsonl schreiben
            with open(profile.train_jsonl_path, "w", encoding="utf-8") as f:
                for clip_data, codes in zip(clips, all_codes):
                    entry = {
                        "audio": clip_data["audio"],
                        "text": clip_data["text"],
                        "ref_audio": str(profile.ref_audio_path),
                        "audio_codes": codes,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Tokenizer entladen (Speicher freigeben für Training)
            await asyncio.to_thread(self.tokenizer.unload)

            self.manager.update_training_status(profile, "data_ready")
            self._toast(f"Training data prepared: {total} clips tokenized!")
            self._reload_profiles(select_id=self.selected_profile_id)

        except Exception as e:
            self._toast(f"Tokenization failed: {e}", error=True)
        finally:
            self.tokenize_progress.visible = False
            self.tokenize_status.value = ""
            self.page.update()

    def _update_tokenize_status(self, msg: str) -> None:
        self.tokenize_status.value = msg
        self.page.update()

    async def _start_training(self, _: ft.ControlEvent) -> None:
        if not self.selected_profile_id:
            self._toast("Select a profile first.", error=True)
            return
        profile = self.manager.get(self.selected_profile_id)
        if not profile:
            return
        if profile.training_status not in ("data_ready", "trained"):
            self._toast("Prepare training data first.", error=True)
            return

        # Engine entladen (Speicher freigeben)
        self.engine.unload_model()

        config = TrainingConfig(
            epochs=int(self.train_epochs_slider.value),
            lr=float(self.train_lr_dropdown.value),
            batch_size=1,
            gradient_accumulation_steps=16,
            force_cpu=self.settings.force_cpu_training,
        )

        # Bestimme resume_epoch falls Logs vorhanden sind
        resume_epoch = None
        if profile.training_log:
            last_epoch = profile.training_log[-1].get("epoch", -1)
            if last_epoch >= 0:
                resume_epoch = last_epoch

        self.active_trainer = VoiceTrainer(profile, config)
        self.manager.update_training_status(profile, "training")
        self.training_progress_bar.visible = True
        self.training_progress_text.value = "Training starting..."
        self.page.update()

        def progress_cb(info: dict):
            epoch = info["epoch"] + 1
            total_epochs = info["total_epochs"]
            step = info["step"] + 1
            total_steps = info["total_steps"]
            loss = info["loss"]
            self.training_progress_text.value = f"Epoch {epoch}/{total_epochs} | Step {step}/{total_steps} | Loss: {loss:.4f}"
            self.training_progress_bar.value = ((epoch - 1) * total_steps + step) / (
                total_epochs * total_steps
            )
            self.page.update()

        def status_cb(msg: str):
            self.training_loss_text.value = msg
            self.page.update()

        try:
            checkpoint_path = await asyncio.to_thread(
                self.active_trainer.train, progress_cb, status_cb, resume_epoch
            )

            if checkpoint_path:
                # Relativen Checkpoint-Namen speichern
                cp_name = Path(checkpoint_path).name
                self.manager.update_training_status(
                    profile,
                    "trained",
                    model_path=cp_name,
                    training_log=self.active_trainer.profile.training_log,
                    training_config=config.to_dict(),
                )
                self._toast("Training complete!")
            else:
                self.manager.update_training_status(profile, "data_ready")
                self._toast("Training cancelled.")

        except Exception as e:
            self.manager.update_training_status(profile, "data_ready")
            self._toast(f"Training failed: {e}", error=True)
        finally:
            self.active_trainer = None
            self.training_progress_bar.visible = False
            self._reload_profiles(select_id=self.selected_profile_id)

    def _stop_training(self, _: ft.ControlEvent) -> None:
        if self.active_trainer:
            self.active_trainer.cancel()
            self.training_progress_text.value = "Stopping..."
            self.page.update()

    async def _load_checkpoint(self, checkpoint_path: str) -> None:
        self._set_top_status("Loading checkpoint...", "warn")
        try:
            await asyncio.to_thread(self.engine.load_finetuned_model, checkpoint_path)
            self._set_top_status("Checkpoint loaded", "ok")
            self._toast("Checkpoint loaded!")
        except Exception as e:
            self._toast(f"Load failed: {e}", error=True)
            self._set_top_status("Error", "error")

    def _delete_checkpoint(self, checkpoint_name: str) -> None:
        if self.selected_profile_id:
            profile = self.manager.get(self.selected_profile_id)
            if profile:
                self.manager.delete_checkpoint(profile, checkpoint_name)
                self._reload_profiles(select_id=self.selected_profile_id)
                self._toast("Checkpoint deleted.")

    def _jump_to_studio(self, _: ft.ControlEvent) -> None:
        if self.selected_profile_id:
            self.speech_profile_dropdown.value = self.selected_profile_id
            self._set_section(2)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: SPEECH STUDIO
    # ══════════════════════════════════════════════════════════════════════
    def _build_speech_studio(self) -> ft.Control:
        self.speech_profile_dropdown = ft.Dropdown(
            label="Trained Profile", border_radius=12
        )
        self.speech_lang_dropdown = ft.Dropdown(
            label="Language",
            options=[ft.dropdown.Option(l) for l in SUPPORTED_LANGUAGES],
            value="German",
        )
        self.speech_text_input = ft.TextField(
            label="Text to speak", multiline=True, min_lines=5, border_radius=12
        )
        self.instruct_input = ft.TextField(
            label="Instruct (emotion/style, optional)",
            hint_text="e.g. 'Speak with excitement and energy'",
            border_radius=12,
        )
        self.generate_status = ft.Text("Ready.", size=12)
        self.generate_progress = ft.ProgressRing(
            width=18, height=18, stroke_width=2, visible=False
        )
        self.output_label = ft.Text("No output.", size=12)

        return self._panel(
            "Speech Studio",
            "Generate speech with your finetuned voice model.",
            [
                ft.Row([self.speech_profile_dropdown, self.speech_lang_dropdown]),
                self.speech_text_input,
                self.instruct_input,
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
        pid = self.speech_profile_dropdown.value
        text = self.speech_text_input.value.strip()
        if not pid or not text:
            self._toast("Select a trained profile and enter text.", error=True)
            return

        profile = self.manager.get(pid)
        if not profile or profile.training_status != "trained":
            self._toast("Profile is not trained.", error=True)
            return

        self.generate_progress.visible = True
        self.generate_status.value = "Generating..."
        self.page.update()

        try:
            # Finetuned Modell laden
            checkpoint_path = str(profile.checkpoints_dir / profile.model_path)
            await asyncio.to_thread(self.engine.load_finetuned_model, checkpoint_path)

            instruct = self.instruct_input.value.strip() or None

            wavs, out_sr = await asyncio.to_thread(
                self.engine.generate_custom_voice,
                text,
                profile.speaker_name,
                self.speech_lang_dropdown.value,
                instruct=instruct,
                max_new_tokens=self.settings.max_token_size,
                temperature=self.settings.temperature,
                repetition_penalty=self.settings.repetition_penalty,
                subtalker_temperature=self.settings.subtalker_temperature,
            )
            combined = np.concatenate(wavs) if isinstance(wavs, list) else wavs
            out_path = (
                profile.exports_dir
                / f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            out_path.parent.mkdir(exist_ok=True)
            await asyncio.to_thread(sf.write, str(out_path), combined, out_sr)
            self.last_output = (combined, out_sr, out_path)
            self.output_label.value = f"Generated: {out_path.name}"
            self._set_top_status("Checkpoint loaded", "ok")
            if self.settings.auto_open_exports:
                self._open_path(out_path)
        except Exception as e:
            self._toast(f"Generation failed: {e}", error=True)
        finally:
            self.generate_progress.visible = False
            self.generate_status.value = "Ready."
            self.page.update()

    def _play_output(self, _: ft.ControlEvent) -> None:
        if self.last_output:
            asyncio.create_task(self._play_clip(str(self.last_output[2])))

    def _stop_output(self, _: ft.ControlEvent) -> None:
        self._stop_playback_process()

    def _save_preview(self, _: ft.ControlEvent) -> None:
        if self.last_output and self.speech_profile_dropdown.value:
            p = self.manager.get(self.speech_profile_dropdown.value)
            if p:
                self.manager.save_preview(p, self.last_output[0], self.last_output[1])
                self._reload_profiles(p.id)
                self._toast("Preview saved.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: SYSTEM SETTINGS
    # ══════════════════════════════════════════════════════════════════════
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
            max=2.5,
            divisions=24,
            label="{value}",
            value=self.settings.temperature,
            on_change=self._save_numeric_settings,
        )
        self.rep_penalty_slider = ft.Slider(
            min=1.0,
            max=2.0,
            divisions=20,
            label="{value}",
            value=self.settings.repetition_penalty,
            on_change=self._save_numeric_settings,
        )
        self.sub_temp_slider = ft.Slider(
            min=0.1,
            max=2.5,
            divisions=24,
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

        self.force_cpu_switch = ft.Switch(
            label="Force CPU Training (slower but stable)",
            value=self.settings.force_cpu_training,
            on_change=self._toggle_force_cpu,
        )

        return self._panel(
            "System Settings",
            "Configure engine, training, and AI parameters.",
            [
                ft.Text("Engine & Device", size=15, weight=ft.FontWeight.W_600),
                ft.Container(
                    padding=16,
                    border_radius=16,
                    bgcolor="#F8FAFC",
                    border=ft.border.all(1, "#E2E8F0"),
                    content=ft.Column(
                        spacing=10,
                        controls=[
                            ft.Row(
                                [
                                    ft.ElevatedButton(
                                        "Preload Tokenizer",
                                        icon=ft.Icons.DOWNLOAD,
                                        on_click=self._preload_tokenizer,
                                    ),
                                    self.model_status_label,
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            ),
                            ft.Text(
                                f"Device: {self.engine.device_label} | "
                                f"Tokenizer: {'Loaded' if self.tokenizer.is_loaded else 'Not loaded'}",
                                size=11,
                                color="#64748B",
                            ),
                            self.force_cpu_switch,
                        ],
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

    # ── System Callbacks ────────────────────────────────────────────────
    def _save_numeric_settings(self, _: ft.ControlEvent) -> None:
        try:
            self.settings.temperature = float(self.temp_slider.value)
            self.settings.repetition_penalty = float(self.rep_penalty_slider.value)
            self.settings.subtalker_temperature = float(self.sub_temp_slider.value)
            self.settings.max_token_size = int(self.max_tokens_input.value or 4092)
            _save_settings(self.settings)
        except ValueError:
            pass

    async def _preload_tokenizer(self, _: ft.ControlEvent) -> None:
        self._set_top_status("Loading tokenizer...", "warn")
        try:
            await asyncio.to_thread(self.tokenizer.load)
            self._refresh_model_status()
            self._toast("Tokenizer loaded!")
        except Exception as e:
            self._toast(f"Load failed: {e}", error=True)
            self._set_top_status("Error", "error")

    def _refresh_model_status(self) -> None:
        parts = []
        if self.engine.model_loaded:
            parts.append(f"Model: {Path(self.engine.current_checkpoint or '').name}")
        if self.tokenizer.is_loaded:
            parts.append("Tokenizer: Loaded")
        status = " | ".join(parts) if parts else "Idle"
        self.model_status_label.value = status
        self._set_top_status(status, "ok" if parts else "idle")

    def _toggle_auto_open(self, e: ft.ControlEvent) -> None:
        self.settings.auto_open_exports = bool(e.control.value)
        _save_settings(self.settings)

    def _toggle_force_cpu(self, e: ft.ControlEvent) -> None:
        self.settings.force_cpu_training = bool(e.control.value)
        _save_settings(self.settings)

    def _save_gemini_api_key(self, e: ft.ControlEvent) -> None:
        self.settings.gemini_api_key = e.control.value
        _save_settings(self.settings)

    def _on_whisper_model_changed(self, e: ft.ControlEvent) -> None:
        self.settings.whisper_model = e.control.value
        _save_settings(self.settings)
        if hasattr(self, "system_whisper_dropdown") and self.system_whisper_dropdown:
            self.system_whisper_dropdown.value = e.control.value
        self.page.update()

    # ── Playback & Utilities ────────────────────────────────────────────
    async def _play_clip(self, path: str) -> None:
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

    def _stop_playback_process(self) -> None:
        if self.playback_process:
            self.playback_process.terminate()
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
        except Exception:
            pass

    def _on_disconnect(self, _: ft.ControlEvent) -> None:
        self._stop_playback_process()


def _main(page: ft.Page) -> None:
    VoiceSynthStudioApp(page)


def run() -> None:
    ft.app(target=_main, view=ft.AppView.FLET_APP, assets_dir="assets")
