"""Test: torch.compile + Profiling der Architektur."""
import os
import time
import torch

def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

log(f"PyTorch {torch.__version__}")

from qwen_tts import Qwen3TTSModel

DEVICE = "mps"
REF_AUDIO = "voices/76fcb15d-8fda-443c-9a18-fd0c989b0728/reference.wav"

# Modell laden
log("Lade Modell…")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map={"": DEVICE},
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
log("Modell geladen.")

# Architektur-Info
talker = model.model.talker
log(f"num_code_groups: {talker.config.num_code_groups}")
log(f"Talker params: {sum(p.numel() for p in talker.parameters()) / 1e6:.0f}M")
log(f"CodePredictor params: {sum(p.numel() for p in talker.code_predictor.parameters()) / 1e6:.0f}M")
log(f"SpeechTokenizer params: {sum(p.numel() for p in model.model.speech_tokenizer.model.parameters()) / 1e6:.0f}M")

# Prompt vorab erstellen um das aus dem Timing rauszunehmen
log("Erstelle Prompt…")
t0 = time.time()
prompt = model.create_voice_clone_prompt(
    ref_audio=REF_AUDIO,
    ref_text="Hello.",
    x_vector_only_mode=True,  # Schneller: nur x-vector, kein ICL
)
torch.mps.synchronize()
log(f"Prompt (x_vector_only) erstellt in {time.time()-t0:.1f}s")

# Baseline ohne compile
log("\n=== Baseline (ohne compile), max_new_tokens=8 ===")
t0 = time.time()
wavs, sr = model.generate_voice_clone(
    text="Hi.",
    language="English",
    voice_clone_prompt=prompt,
    max_new_tokens=8,
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
t = time.time() - t0
log(f"Baseline: {t:.1f}s für {wavs[0].shape[0]/sr:.1f}s Audio")
log(f"  → {8/t:.2f} talker-tok/s, {8*16/t:.2f} total-tok/s")

# Mit torch.compile
log("\n=== torch.compile (inductor) ===")
try:
    talker.model = torch.compile(talker.model, backend="aot_eager")
    log("Talker model compiled (aot_eager)")

    # Warmup
    log("Warmup…")
    t0 = time.time()
    wavs2, sr2 = model.generate_voice_clone(
        text="Hi.",
        language="English",
        voice_clone_prompt=prompt,
        max_new_tokens=4,
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
    log(f"Warmup: {time.time()-t0:.1f}s")

    # Actual test
    log("Compiled test, max_new_tokens=8…")
    t0 = time.time()
    wavs3, sr3 = model.generate_voice_clone(
        text="Hi.",
        language="English",
        voice_clone_prompt=prompt,
        max_new_tokens=8,
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
    t = time.time() - t0
    log(f"Compiled: {t:.1f}s für {wavs3[0].shape[0]/sr3:.1f}s Audio")
    log(f"  → {8/t:.2f} talker-tok/s")
except Exception as e:
    log(f"torch.compile fehlgeschlagen: {e}")
