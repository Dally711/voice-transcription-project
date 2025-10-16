import sounddevice as sd
import numpy as np
import tkinter as tk
import threading
import subprocess
import sys
from faster_whisper import WhisperModel

# --- Settings ---
LANG = "en"          # "fr" for French
MODEL = "base"       # Options: tiny, base, small, medium, large
FS = 16000           # Sampling rate
DURATION = 5         # Duration of recording in seconds

# --- Load Whisper model ---
print("Loading Whisper model...")
model = WhisperModel(MODEL, device="cpu")
print("Model loaded successfully.")

# --- Tkinter GUI setup ---
root = tk.Tk()
root.title("Voice Transcriber and Speaker")
root.geometry("800x500")
root.configure(bg="#f8f8f8")

title = tk.Label(root, text=f"Voice Transcriber ({LANG.upper()})",
                 font=("Arial", 18, "bold"), bg="#f8f8f8")
title.pack(pady=10)

text_box = tk.Text(root, wrap=tk.WORD, font=("Arial", 14),
                   height=15, bg="white")
text_box.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

status_label = tk.Label(root, text="Status: Idle",
                        font=("Arial", 12), bg="#f8f8f8")
status_label.pack(pady=5)

# --- Record + Transcribe + Speak ---
def record_and_transcribe():
    status_label.config(text="Status: Recording...")
    text_box.delete(1.0, tk.END)

    # Record audio
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype=np.float32)
    sd.wait()

    status_label.config(text="Status: Transcribing...")

    # Transcribe using Whisper
    audio = audio.flatten()
    segments, _ = model.transcribe(audio, language=LANG)

    result_text = ""
    for seg in segments:
        result_text += seg.text.strip() + " "

    if result_text:
        text_box.insert(tk.END, result_text)
        text_box.see(tk.END)
        status_label.config(text="Status: Speaking...")

        # Speak text using a new process each time
        try:
            subprocess.run(
                [sys.executable, "tts_once.py"],
                input=result_text,
                text=True,
                check=False
            )
        except Exception as e:
            print("Speech error:", e)

    status_label.config(text="Status: Done")

# --- Start button ---
def start_transcription():
    threading.Thread(target=record_and_transcribe, daemon=True).start()

# --- Button design ---
button_frame = tk.Frame(root, bg="#f8f8f8")
button_frame.pack(pady=(0, 10))

start_button = tk.Button(root,
                         text="Start",
                         font=("Arial", 16, "bold"),  # Larger text
                         command=start_transcription,
                         bg="#4CAF50", fg="white",
                         width=10, height=3)  # size
start_button.pack(ipady=10)

root.mainloop()
