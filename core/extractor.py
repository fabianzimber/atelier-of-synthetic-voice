import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import numpy as np
import soundfile as sf
import torch
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
    clarity_score: int
    duration: float


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

                # Skip very short or very long clips
                if duration < 2.0 or duration > 60.0:
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
                                "Listen to this audio clip. "
                                "1. Transcribe the spoken text exactly. "
                                "2. Rate the audio quality and emotional expressiveness from 1 to 10. "
                                "BE CRITICAL: 10 is perfect studio quality with very clear, distinct strong emotion. "
                                "8 is high quality but might have slight noise or very subtle/neutral emotion. "
                                "Only give 9 or 10 for exceptionally clear clips where the emotion is vivid and unmistakable. "
                                "Neutral speech should be rated strictly lower unless it is perfectly clean. "
                                "3. Classify the emotion (e.g. 'neutral', 'angry', 'happy', 'sad', 'excited', 'whispering'). "
                                "Return ONLY a valid JSON object with keys 'transcript', 'clarity_score' (int), and 'emotion' (string)."
                            )

                            myfile = await asyncio.to_thread(
                                self.client.files.upload, file=tmp_chunk_path
                            )
                            response = await asyncio.to_thread(
                                self.client.models.generate_content,
                                model="gemini-2.5-flash",
                                contents=[myfile, prompt],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json"
                                ),
                            )

                            data = json.loads(response.text)
                            clarity = int(data.get("clarity_score", 0))

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
