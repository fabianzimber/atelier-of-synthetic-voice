"""Test: MPS fallback warnings + alternative Lademethode."""
import os
import time
import torch
import numpy as np

# MPS Fallback-Warnungen aktivieren
os.environ["PYTORCH_MPS_FALLBACK_POLICY"] = "warn"

def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

log("Import…")
from qwen_tts import Qwen3TTSModel

DEVICE = "mps"
log(f"Device: {DEVICE}, PyTorch: {torch.__version__}")
log(f"MPS available: {torch.backends.mps.is_available()}")

# Modell laden
log("Lade Modell…")
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map={"": DEVICE},
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
log(f"Modell geladen in {time.time()-t0:.1f}s")

# Kurzer Test
REF_AUDIO = "voices/76fcb15d-8fda-443c-9a18-fd0c989b0728/reference.wav"
TEXT = "Hi."

log("Generiere kurzen Text (max_new_tokens=32)…")
t0 = time.time()
wavs, sr = model.generate_voice_clone(
    text=TEXT,
    language="English",
    ref_audio=REF_AUDIO,
    ref_text="Hello.",
    max_new_tokens=32,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
)
torch.mps.synchronize()
log(f"Fertig in {time.time()-t0:.1f}s → {wavs[0].shape[0]/sr:.1f}s Audio")
