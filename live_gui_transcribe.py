import tkinter as tk
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import pyttsx3


# --- SETTINGS ---
LANG = "en"          # or "fr"
MODEL_SIZE = "base"  # tiny, base, small, medium, large
FS = 16000           # Sampling rate
DURATION = 5         # Recording length (seconds)


# --- LOAD WHISPER MODEL ---
print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE)
print("Model loaded successfully.")


# --- RECORD AUDIO ---
def record_audio(duration=DURATION, fs=FS):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    return np.squeeze(audio)


# --- TRANSCRIBE AUDIO ---
def transcribe_audio(audio):
    print("Transcribing...")
    segments, _ = model.transcribe(audio, beam_size=5, language=LANG)
    text = " ".join([segment.text for segment in segments])
    print("Transcription:", text)
    return text


# --- SPEAK TEXT ---
def speak_text(text):
    if not text:
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


# --- GUI SETUP ---
root = tk.Tk()
root.title("Voice Transcriber")
root.geometry("800x500")
root.configure(bg="#f8f8f8")

title = tk.Label(
    root,
    text=f"Voice Transcriber ({LANG.upper()})",
    font=("Arial", 18, "bold"),
    bg="#f8f8f8"
)
title.pack(pady=10)

text_box = tk.Text(
    root,
    wrap=tk.WORD,
    font=("Arial", 14),
    height=15,
    bg="white"
)
text_box.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

status_label = tk.Label(
    root,
    text="Status: Idle",
    font=("Arial", 12),
    bg="#f8f8f8"
)
status_label.pack(pady=5)


# --- MAIN FUNCTION ---
def record_and_transcribe():
    status_label.config(text="Status: Recording...")
    text_box.delete(1.0, tk.END)
    audio = record_audio()

    status_label.config(text="Status: Transcribing...")
    result_text = transcribe_audio(audio)

    if result_text:
        text_box.insert(tk.END, result_text)
        text_box.update_idletasks()
        status_label.config(text="Status: Speaking...")
        speak_text(result_text)

    status_label.config(text="Status: Done")


def start_transcription():
    threading.Thread(target=record_and_transcribe, daemon=True).start()


# --- BUTTON ---
start_button = tk.Button(
    root,
    text="Start",
    font=("Arial", 18, "bold"),
    command=start_transcription,
    bg="#4CAF50",
    fg="white",
    width=10,
)
start_button.pack(pady=20, ipady=15)

root.mainloop()
