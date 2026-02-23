"""Detaillierterer Test: Timing pro Phase + Output-Analyse."""
import time
import torch
import numpy as np
import soundfile as sf

def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

log("Import…")
from qwen_tts import Qwen3TTSModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
log(f"Device: {DEVICE}")

# Modell laden
log("Lade Modell…")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map={"": DEVICE},
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
log("Modell geladen.")

# Prüfe alle Submodule auf Device
for name, param in model.model.named_parameters():
    if str(param.device) != "mps:0":
        log(f"  WARNUNG: {name} auf {param.device}!")
        break
else:
    log("Alle Parameter auf mps:0 ✓")

# Prüfe Speech-Tokenizer separat
for name, param in model.model.speech_tokenizer.model.named_parameters():
    if str(param.device) != "mps:0":
        log(f"  WARNUNG: speech_tokenizer.{name} auf {param.device}!")
        break
else:
    log("Speech-Tokenizer alle Parameter auf mps:0 ✓")

# Referenz-Audio
REF_AUDIO = "voices/76fcb15d-8fda-443c-9a18-fd0c989b0728/reference.wav"
REF_TEXT = "Hello, this is a test."
TEXT = "Hello world."

# Manuell die Schritte aufteilen um zu timen
log("=== Phase 1: create_voice_clone_prompt ===")
t0 = time.time()
prompt = model.create_voice_clone_prompt(
    ref_audio=REF_AUDIO,
    ref_text=REF_TEXT,
    x_vector_only_mode=False,
)
if DEVICE == "mps":
    torch.mps.synchronize()
log(f"Prompt erstellt in {time.time()-t0:.1f}s")

# Phase 2: generate mit kleinem max_new_tokens
log("=== Phase 2: generate_voice_clone (max_new_tokens=128) ===")
t0 = time.time()
wavs, sr = model.generate_voice_clone(
    text=TEXT,
    language="Auto",
    voice_clone_prompt=prompt,
    max_new_tokens=128,
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
if DEVICE == "mps":
    torch.mps.synchronize()
gen_time = time.time() - t0
audio_len = wavs[0].shape[0] / sr if wavs else 0
log(f"Fertig in {gen_time:.1f}s → {audio_len:.1f}s Audio, shape={wavs[0].shape}")

sf.write("test_output2.wav", wavs[0], sr)
log("Gespeichert als test_output2.wav")

# Waveform stats
w = wavs[0]
log(f"Waveform: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}, std={w.std():.4f}")
# Prüfe ob es Stille ist
rms = np.sqrt(np.mean(w**2))
log(f"RMS: {rms:.6f} {'(Stille!)' if rms < 0.001 else '(hat Audio)'}")
