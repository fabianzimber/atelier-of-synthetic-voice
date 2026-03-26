import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
import torchaudio

# Monkey patch torchaudio.list_audio_backends for newer torchaudio versions used by speechbrain
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# Suppress speechbrain's use of deprecated torch.cuda.amp.custom_fwd (fixed in PyTorch 2.x)
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.custom_fwd.*",
    category=FutureWarning,
    module="speechbrain",
)

from google import genai
from google.genai import types
from speechbrain.inference.speaker import SpeakerRecognition


@dataclass
class ExtractedClip:
    path: str
    transcript: str
    emotion: str
    clarity_score: int       # Weighted composite (1–10)
    duration: float
    speaker_similarity: float = 0.0
    audio_quality: int = 0   # Sub-score: recording quality (1–10)
    expressiveness: int = 0  # Sub-score: emotional dynamics (1–10)
    speech_clarity: int = 0  # Sub-score: articulation / intelligibility (1–10)


class AudioExtractor:
    def __init__(
        self,
        target_audio_path: str,
        source_media_path: str,
        output_dir: str,
        gemini_api_key: Optional[str] = None,
    ):
        self.target_audio_path = Path(target_audio_path)
        self.source_media_path = Path(source_media_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gemini_api_key = gemini_api_key

        self.vad_model = None
        self.get_speech_timestamps = None
        self.speaker_model = None
        self.client = None
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def _load_models(self, status_cb: Callable[[str], None]):
        status_cb("Loading Silero VAD (Voice Activity Detection)...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        self.get_speech_timestamps = utils[0]

        status_cb("Loading SpeechBrain (Speaker Verification)...")
        # Ensure we have a cache dir
        cache_dir = Path.home() / ".cache" / "voice_clone_extractor"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(cache_dir / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"},
        )

        if self.gemini_api_key:
            status_cb("Initializing Gemini 2.5 API...")
            self.client = genai.Client(api_key=self.gemini_api_key)

    def _extract_audio(self, status_cb: Callable[[str], None]) -> Path:
        status_cb("Extracting audio from media (16kHz, mono)...")
        # Use a unique temporary file to avoid collisions between multiple runs
        fd, path_str = tempfile.mkstemp(suffix="_source.wav")
        os.close(fd)
        tmp_wav = Path(path_str)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(self.source_media_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(tmp_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if tmp_wav.exists():
                tmp_wav.unlink()
            raise RuntimeError(f"FFmpeg extraction failed: {result.stderr}")
        return tmp_wav

    async def extract(
        self, status_cb: Callable[[str], None], stats_cb: Callable[[dict], None]
    ) -> AsyncGenerator[ExtractedClip, None]:
        await asyncio.to_thread(self._load_models, status_cb)

        tmp_wav = await asyncio.to_thread(self._extract_audio, status_cb)

        try:
            status_cb("Analyzing target voice signature...")
            target_sig = await asyncio.to_thread(
                self.speaker_model.load_audio, str(self.target_audio_path)
            )

            status_cb("Extraction in progress. Searching for perfect clips...")

            wav, sr = await asyncio.to_thread(sf.read, str(tmp_wav), dtype="float32")
            wav_tensor = torch.from_numpy(wav)

            status_cb("Running Voice Activity Detection...")
            speech_timestamps = await asyncio.to_thread(
                self.get_speech_timestamps,
                wav_tensor,
                self.vad_model,
                sampling_rate=16000,
            )

            stats = {
                "total": len(speech_timestamps),
                "processed": 0,
                "rejected_speaker": 0,
                "rejected_quality": 0,
                "found": 0,
            }
            stats_cb(stats)

            for i, ts in enumerate(speech_timestamps):
                if self.is_cancelled:
                    break

                start = ts["start"]
                end = ts["end"]
                duration = (end - start) / 16000.0

                # Skip very short or very long clips (optimal for zero-shot cloning: 5–25 s)
                if duration < 5.0 or duration > 25.0:
                    stats["processed"] += 1
                    stats_cb(stats)
                    continue

                chunk_tensor = wav_tensor[start:end].unsqueeze(0)

                # Speaker Verification
                score, _ = await asyncio.to_thread(
                    self.speaker_model.verify_batch, target_sig, chunk_tensor
                )
                similarity = score.item()

                if similarity < 0.50:
                    stats["rejected_speaker"] += 1
                    stats["processed"] += 1
                    stats_cb(stats)
                    continue

                # It's our speaker! Save a temp file to send to Gemini
                chunk_np = wav_tensor[start:end].numpy()
                fd, tmp_chunk_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)

                try:
                    await asyncio.to_thread(sf.write, tmp_chunk_path, chunk_np, 16000)

                    clip = None
                    if self.client:
                        status_cb(
                            f"Gemini evaluation for chunk {i + 1}/{len(speech_timestamps)}..."
                        )
                        try:
                            prompt = (
                                "You are a strict audio quality auditor for voice cloning. "
                                "Evaluate this clip on THREE independent dimensions, each scored 1–10:\n\n"
                                "audio_quality — Recording technical quality:\n"
                                "  10: Studio silence, zero noise floor, pristine\n"
                                "   8: Very clean, barely perceptible room tone\n"
                                "   6: Noticeable hiss, mild background sound, light compression\n"
                                "   4: Significant noise, music bed, or competing voices audible\n"
                                "   2: Heavy distortion, clipping, or near-unintelligible noise\n\n"
                                "voice_expressiveness — Emotional range and prosody dynamics:\n"
                                "  10: Vivid unmistakable emotion dominating the clip, strong pitch/rhythm dynamics\n"
                                "   8: Clear emotional coloring, natural prosody variation\n"
                                "   6: Some inflection, but mostly flat; serviceable but unremarkable\n"
                                "   4: Monotone throughout, robotic or completely neutral pacing\n"
                                "   2: No discernible expressiveness whatsoever\n\n"
                                "speech_clarity — Intelligibility and articulation:\n"
                                "  10: Every word perfectly crisp, ideal for forced-alignment transcription\n"
                                "   8: Very clear, minor articulation inconsistencies\n"
                                "   6: Mostly understandable, occasional mumbling or fast-speech drops\n"
                                "   4: Frequently hard to follow, many unclear phonemes\n"
                                "   2: Largely unintelligible\n\n"
                                "IMPORTANT: Use the FULL scale. Most clips should score 5–8. "
                                "Only exceptional clips reach 9–10. Be strict and differentiate clearly. "
                                "Do NOT default everything to 8.\n\n"
                                "Also:\n"
                                "1. Transcribe the spoken text exactly.\n"
                                "2. Classify the dominant emotion/style "
                                "(e.g. 'neutral', 'angry', 'happy', 'sad', 'excited', 'whispering', 'narrative', 'dramatic').\n\n"
                                "Return ONLY valid JSON with keys: "
                                "'transcript' (string), 'audio_quality' (int), "
                                "'voice_expressiveness' (int), 'speech_clarity' (int), 'emotion' (string)."
                            )

                            myfile = await asyncio.to_thread(
                                self.client.files.upload, file=tmp_chunk_path
                            )
                            response = await asyncio.to_thread(
                                self.client.models.generate_content,
                                model="gemini-3-flash-preview",
                                contents=[myfile, prompt],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json"
                                ),
                            )

                            data = json.loads(response.text)
                            # Weighted composite: expressiveness 40%, audio quality 35%, clarity 25%
                            aq = int(data.get("audio_quality", 5))
                            ve = int(data.get("voice_expressiveness", 5))
                            sc = int(data.get("speech_clarity", 5))
                            clarity = round(aq * 0.35 + ve * 0.40 + sc * 0.25)

                            if clarity >= 8:
                                emotion_clean = (
                                    str(data.get("emotion", "unknown"))
                                    .replace(" ", "_")
                                    .replace("/", "_")
                                )
                                out_name = f"extracted_{emotion_clean}_score{clarity}_{start}.wav"
                                out_path = self.output_dir / out_name
                                await asyncio.to_thread(
                                    shutil.copy, tmp_chunk_path, out_path
                                )

                                clip = ExtractedClip(
                                    path=str(out_path),
                                    transcript=data.get("transcript", ""),
                                    emotion=data.get("emotion", "unknown"),
                                    clarity_score=clarity,
                                    duration=duration,
                                    speaker_similarity=similarity,
                                    audio_quality=aq,
                                    expressiveness=ve,
                                    speech_clarity=sc,
                                )
                            else:
                                stats["rejected_quality"] += 1

                        except Exception as e:
                            print(f"Gemini error on chunk {i}:", e)
                            # Fallback if Gemini fails
                            out_name = f"extracted_clip_{start}.wav"
                            out_path = self.output_dir / out_name
                            await asyncio.to_thread(
                                shutil.copy, tmp_chunk_path, out_path
                            )
                            clip = ExtractedClip(
                                path=str(out_path),
                                transcript="",
                                emotion="unclassified_api_error",
                                clarity_score=0,
                                duration=duration,
                                speaker_similarity=similarity,
                            )

                    if not self.client:
                        out_name = f"extracted_clip_{start}.wav"
                        out_path = self.output_dir / out_name
                        await asyncio.to_thread(shutil.copy, tmp_chunk_path, out_path)
                        clip = ExtractedClip(
                            path=str(out_path),
                            transcript="",
                            emotion="unclassified",
                            clarity_score=0,
                            duration=duration,
                            speaker_similarity=similarity,
                        )

                    stats["processed"] += 1
                    if clip:
                        stats["found"] += 1
                        status_cb(
                            f"Found new clip: {clip.emotion} (Score: {clip.clarity_score})"
                        )
                        yield clip
                    stats_cb(stats)

                finally:
                    if os.path.exists(tmp_chunk_path):
                        os.remove(tmp_chunk_path)

        finally:
            if tmp_wav.exists():
                tmp_wav.unlink()
            status_cb(
                "Extraction finished."
                if not self.is_cancelled
                else "Extraction cancelled."
            )

    @staticmethod
    def build_optimal_reference(
        clips: list[ExtractedClip],
        target_dir: Path,
        target_duration: float = 28.0,
        silence_gap: float = 0.3,
    ) -> tuple[Path, str] | None:
        """Build an optimal voice cloning reference audio from extracted clips.

        Selects and concatenates clips (sorted by clarity_score * speaker_similarity)
        until target_duration is reached (~28 s for maximum emotional variety).
        A single clip >= 25 s with score >= 8 is used directly without merging.

        Returns (wav_path, combined_transcript) or None if no clips available.
        """
        if not clips:
            return None

        # Sort by composite score descending
        scored = sorted(
            clips,
            key=lambda c: c.clarity_score * c.speaker_similarity,
            reverse=True,
        )

        # Only skip accumulation if a single clip is already near the 30 s target
        for c in scored:
            if c.duration >= 25.0 and c.clarity_score >= 8:
                return Path(c.path), c.transcript

        # Otherwise accumulate clips until target_duration
        selected: list[ExtractedClip] = []
        accumulated = 0.0
        for c in scored:
            if accumulated >= target_duration:
                break
            selected.append(c)
            accumulated += c.duration + silence_gap

        if not selected:
            return None

        if len(selected) == 1:
            return Path(selected[0].path), selected[0].transcript

        # Load and concatenate audio segments
        segments: list[np.ndarray] = []
        sr: int = 24000
        silence = np.zeros(int(sr * silence_gap), dtype=np.float32)

        for clip in selected:
            audio, clip_sr = sf.read(clip.path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if clip_sr != sr:
                g = gcd(sr, clip_sr)
                audio = resample_poly(audio, sr // g, clip_sr // g).astype(np.float32)
            segments.append(audio)
            segments.append(silence)

        combined = np.concatenate(segments[:-1])  # Drop trailing silence
        combined_transcript = " ".join(c.transcript for c in selected if c.transcript)

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "optimal_reference.wav"
        sf.write(str(out_path), combined, sr)

        return out_path, combined_transcript

    @staticmethod
    def filter_top_per_emotion(
        clips: list[ExtractedClip], max_per_emotion: int = 3
    ) -> list[ExtractedClip]:
        """Keep only the top N clips per emotion (by clarity_score), delete the rest from disk."""
        from collections import defaultdict

        by_emotion: dict[str, list[ExtractedClip]] = defaultdict(list)
        for clip in clips:
            by_emotion[clip.emotion].append(clip)

        keep: list[ExtractedClip] = []
        for emotion, group in by_emotion.items():
            group.sort(key=lambda c: c.clarity_score, reverse=True)
            for clip in group[max_per_emotion:]:
                p = Path(clip.path)
                if p.exists():
                    p.unlink()
            keep.extend(group[:max_per_emotion])

        return keep
