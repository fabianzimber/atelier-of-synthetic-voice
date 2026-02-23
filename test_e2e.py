"""End-to-End Test mit TTSEngine – mit Timeout."""
import signal
import time
import soundfile as sf

class Timeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise Timeout("Timeout!")

def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

from core.engine import TTSEngine

engine = TTSEngine()
REF_AUDIO = "voices/76fcb15d-8fda-443c-9a18-fd0c989b0728/reference.wav"

# Modell laden
log("=== Modell laden ===")
t0 = time.time()
engine.load_clone_model(progress_cb=log)
log(f"Load: {time.time()-t0:.1f}s")

# Test mit 120s Timeout
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(120)

try:
    log("\n=== Generierung (max 120s) ===")
    t0 = time.time()
    wavs, sr = engine.generate_with_clone(
        text="Hello, how are you?",
        language="English",
        ref_audio=REF_AUDIO,
        ref_text="Hello, this is a test.",
        progress_cb=log,
    )
    signal.alarm(0)  # Cancel timeout
    gen_time = time.time() - t0
    audio_len = wavs[0].shape[0] / sr
    log(f"Ergebnis: {gen_time:.1f}s → {audio_len:.1f}s Audio (RTF: {gen_time/audio_len:.2f}x)")
    sf.write("test_e2e_output.wav", wavs[0], sr)
    log("Gespeichert als test_e2e_output.wav")
except Timeout:
    signal.alarm(0)
    log(f"TIMEOUT nach 120s! Generierung hängt.")

log("\n=== Fertig ===")
