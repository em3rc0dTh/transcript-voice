import sounddevice as sd
import threading
from faster_whisper import WhisperModel
import torch
import queue
import time
import numpy as np
import os

# ===== CONFIG =====
# ==Change to medium for better accuracy==
WHISPER_MODEL = "turbo"
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"

AUDIO_QUEUE = queue.Queue(maxsize=6)
STOP_FLAG = threading.Event()


# =====================================================
# 🧩 Worker for real-time transcription
# =====================================================
def transcription_worker(model: WhisperModel):
    """Continuously transcribes incoming audio chunks from the queue."""
    print("🧠 Transcription worker started.")
    while not STOP_FLAG.is_set():
        try:
            audio_chunk = AUDIO_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            segments, info = model.transcribe(
                audio_chunk,
                beam_size=1,
                language="es",
                vad_filter=True,
                condition_on_previous_text=False,
            )
            for seg in segments:
                text = seg.text.strip()
                if text:
                    print(f"[{seg.start:.1f} → {seg.end:.1f}] {text}")
        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
        AUDIO_QUEUE.task_done()


# =====================================================
# 🎙️ Microphone capture for real-time mode
# =====================================================
def audio_callback(indata, frames, time_info, status):
    """Pushes incoming audio data to the queue for processing."""
    if status:
        print("⚠️", status)
    try:
        AUDIO_QUEUE.put_nowait(np.copy(indata[:, 0]))
    except queue.Full:
        print("⚠️ Audio queue full — chunk dropped")


def realtime_transcription():
    """Handles real-time microphone transcription."""
    print("🎙️ Loading Whisper model...")
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"✅ Model loaded on {DEVICE}")

    worker = threading.Thread(target=transcription_worker, args=(model,), daemon=True)
    worker.start()

    print("🎧 Listening... (Ctrl+C to stop)")
    blocksize = int(SAMPLE_RATE * CHUNK_DURATION)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=audio_callback,
        ):
            while not STOP_FLAG.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        STOP_FLAG.set()
    finally:
        AUDIO_QUEUE.join()
        print("✅ Gracefully stopped.")


# =====================================================
# 🎞️ Offline audio file transcription (MP3, WAV, etc.)
# =====================================================
def file_transcription(audio_path):
    """Transcribes an audio file (MP3, WAV, etc.) and saves text output."""
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return

    print(f"🎧 Transcribing file: {audio_path}")
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        language="es",
        vad_filter=True,
        condition_on_previous_text=False,
    )

    output_file = os.path.splitext(audio_path)[0] + "_transcript.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for seg in segments:
            line = f"[{seg.start:.1f} → {seg.end:.1f}] {seg.text.strip()}"
            print(line)
            f.write(line + "\n")

    print(f"✅ Transcription saved as {output_file}")


# =====================================================
# 🚀 Entry point
# =====================================================
if __name__ == "__main__":
    mode = input("Select mode: (1) Real-time mic  (2) File transcription → ").strip()
    if mode == "1":
        realtime_transcription()
    elif mode == "2":
        audio_path = input("Enter path to audio file (e.g., audio.mp3): ").strip()
        file_transcription(audio_path)
    else:
        print("❌ Invalid option. Choose 1 or 2.")
