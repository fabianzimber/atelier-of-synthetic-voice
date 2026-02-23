"""Minimaler Inferenz-Test mit Timing pro Schritt."""
import time
import torch
import numpy as np

def log(msg):
    print(f"[{time.time():.1f}] {msg}")

log("Import qwen_tts…")
from qwen_tts import Qwen3TTSModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
log(f"Device: {DEVICE}")

# 1) Modell laden
log("Lade Modell…")
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map={"": DEVICE},
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
log(f"Modell geladen in {time.time()-t0:.1f}s")

# 2) Prüfe Devices
main_dev = next(model.model.parameters()).device
st_dev = model.model.speech_tokenizer.device
log(f"Hauptmodell device: {main_dev}")
log(f"Speech-Tokenizer device: {st_dev}")

# Speech-Tokenizer auf MPS schieben falls nötig
if str(st_dev) != DEVICE:
    log(f"Speech-Tokenizer auf {DEVICE} verschieben…")
    t0 = time.time()
    model.model.speech_tokenizer.model = model.model.speech_tokenizer.model.to(DEVICE)
    model.model.speech_tokenizer.device = torch.device(DEVICE)
    log(f"Verschoben in {time.time()-t0:.1f}s")

# Nochmal prüfen
st_dev2 = model.model.speech_tokenizer.device
st_param_dev = next(model.model.speech_tokenizer.model.parameters()).device
log(f"Speech-Tokenizer device nach fix: {st_dev2}, param device: {st_param_dev}")

# 3) Generieren
REF_AUDIO = "voices/76fcb15d-8fda-443c-9a18-fd0c989b0728/reference.wav"
REF_TEXT = "Hello, this is a test."
TEXT = "This is a short test sentence."

log("Starte generate_voice_clone…")
t0 = time.time()
wavs, sr = model.generate_voice_clone(
    text=TEXT,
    language="Auto",
    ref_audio=REF_AUDIO,
    ref_text=REF_TEXT,
    max_new_tokens=512,
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
log(f"Generierung fertig in {gen_time:.1f}s")
log(f"Output: {len(wavs)} wav(s), sr={sr}, shape={wavs[0].shape if wavs else 'N/A'}")

# Speichern
import soundfile as sf
sf.write("test_output.wav", wavs[0], sr)
log("Gespeichert als test_output.wav")
