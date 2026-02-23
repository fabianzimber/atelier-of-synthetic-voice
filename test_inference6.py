"""End-to-End Test mit der TTSEngine inkl. torch.compile."""
import time
import soundfile as sf

def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

from core.engine import TTSEngine

engine = TTSEngine()

REF_AUDIO = "voices/76fcb15d-8fda-443c-9a18-fd0c989b0728/reference.wav"

# 1) Modell laden (inkl. compile)
log("=== Modell laden ===")
t0 = time.time()
engine.load_clone_model(progress_cb=log)
log(f"Total Load: {time.time()-t0:.1f}s")

# 2) Prompt erstellen
log("\n=== Prompt erstellen ===")
t0 = time.time()
prompt = engine.create_voice_clone_prompt(
    ref_audio=REF_AUDIO,
    ref_text="Hello, this is a test.",
    progress_cb=log,
)
log(f"Prompt: {time.time()-t0:.1f}s")

# 3) Warmup (compile overhead)
log("\n=== Warmup (compile) ===")
t0 = time.time()
wavs, sr = engine.generate_with_prompt(
    text="Hi.",
    language="English",
    voice_clone_prompt=prompt,
    progress_cb=log,
)
log(f"Warmup: {time.time()-t0:.1f}s")

# 4) Reale Generierung - kurzer Satz
log("\n=== Kurzer Satz ===")
t0 = time.time()
wavs, sr = engine.generate_with_prompt(
    text="Good morning, how are you doing today?",
    language="English",
    voice_clone_prompt=prompt,
    progress_cb=log,
)
gen_time = time.time() - t0
audio_len = wavs[0].shape[0] / sr
log(f"Kurzer Satz: {gen_time:.1f}s → {audio_len:.1f}s Audio (RTF: {gen_time/audio_len:.2f}x)")
sf.write("test_output_short.wav", wavs[0], sr)

# 5) Reale Generierung - längerer Satz
log("\n=== Längerer Satz ===")
t0 = time.time()
wavs, sr = engine.generate_with_prompt(
    text="The quick brown fox jumps over the lazy dog. This is a longer sentence to test the generation speed with a more realistic text input.",
    language="English",
    voice_clone_prompt=prompt,
    progress_cb=log,
)
gen_time = time.time() - t0
audio_len = wavs[0].shape[0] / sr
log(f"Langer Satz: {gen_time:.1f}s → {audio_len:.1f}s Audio (RTF: {gen_time/audio_len:.2f}x)")
sf.write("test_output_long.wav", wavs[0], sr)

# 6) Ohne vorbereiteten Prompt (alles inkl.)
log("\n=== Direkt ohne Prompt-Cache ===")
t0 = time.time()
wavs, sr = engine.generate_with_clone(
    text="Hello world, this is a test.",
    language="English",
    ref_audio=REF_AUDIO,
    ref_text="Hello, this is a test.",
    progress_cb=log,
)
gen_time = time.time() - t0
audio_len = wavs[0].shape[0] / sr
log(f"Direkt: {gen_time:.1f}s → {audio_len:.1f}s Audio (RTF: {gen_time/audio_len:.2f}x)")
sf.write("test_output_direct.wav", wavs[0], sr)

log("\n=== Fertig ===")
