"""Test: Verschiedene Lade-Strategien vergleichen."""
import os
import time
import torch

def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

log(f"PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}")

from qwen_tts import Qwen3TTSModel

REF_AUDIO = "voices/76fcb15d-8fda-443c-9a18-fd0c989b0728/reference.wav"

def test_model(label, model):
    """Kurzer Generierungstest."""
    log(f"\n{'='*40}")
    log(f"Test: {label}")

    # Check device
    p = next(model.model.parameters())
    log(f"  Model device: {p.device}, dtype: {p.dtype}")
    st_p = next(model.model.speech_tokenizer.model.parameters())
    log(f"  SpeechTok device: {st_p.device}, dtype: {st_p.dtype}")

    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text="Hi.",
        language="English",
        ref_audio=REF_AUDIO,
        ref_text="Hello.",
        max_new_tokens=16,
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
    log(f"  → {t:.1f}s für {wavs[0].shape[0]/sr:.1f}s Audio")
    del model
    torch.mps.empty_cache()

# Test 1: device_map + sdpa (aktuelle Methode)
log("\n=== Test 1: device_map={'': 'mps'} + sdpa ===")
t0 = time.time()
m1 = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map={"": "mps"},
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
log(f"Geladen in {time.time()-t0:.1f}s")
test_model("device_map + sdpa", m1)

# Test 2: device_map + eager
log("\n=== Test 2: device_map={'': 'mps'} + eager ===")
t0 = time.time()
m2 = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map={"": "mps"},
    dtype=torch.bfloat16,
    attn_implementation="eager",
)
log(f"Geladen in {time.time()-t0:.1f}s")
test_model("device_map + eager", m2)

# Test 3: CPU laden, dann .to(mps)
log("\n=== Test 3: CPU laden → .to('mps') + eager ===")
t0 = time.time()
m3 = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    dtype=torch.bfloat16,
    attn_implementation="eager",
)
m3.model = m3.model.to("mps")
m3.device = torch.device("mps")
m3.model.speech_tokenizer.model = m3.model.speech_tokenizer.model.to("mps")
m3.model.speech_tokenizer.device = torch.device("mps")
log(f"Geladen + verschoben in {time.time()-t0:.1f}s")
test_model("CPU→MPS + eager", m3)
