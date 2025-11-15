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

stop_event = threading.Event()
is_recording = False


# --- RECORD AUDIO ---
def record_audio(stop_event, duration=DURATION, fs=FS):
    print("Recording...")
    audio_chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_chunks.append(indata.copy())
        if stop_event.is_set():
            raise sd.CallbackStop

    max_duration = duration if duration else DURATION
    try:
        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            stop_event.wait(timeout=max_duration)
    except sd.CallbackStop:
        pass

    print("Recording finished.")
    if not audio_chunks:
        return np.array([], dtype="float32")
    audio = np.concatenate(audio_chunks, axis=0)
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
    global is_recording
    is_recording = True
    status_label.config(text="Status: Recording...")
    text_box.delete(1.0, tk.END)
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

    try:
        audio = record_audio(stop_event)
        stop_button.config(state=tk.DISABLED)
        was_stopped = stop_event.is_set()
        stop_event.clear()

        if audio.size == 0:
            status_text = "Status: Recording stopped" if was_stopped else "Status: No audio captured"
            status_label.config(text=status_text)
            return

        status_label.config(text="Status: Transcribing...")
        result_text = transcribe_audio(audio)

        if result_text:
            text_box.insert(tk.END, result_text)
            text_box.update_idletasks()
            status_label.config(text="Status: Speaking...")
            speak_text(result_text)

        status_label.config(text="Status: Done")
    finally:
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        is_recording = False


def start_transcription():
    global is_recording
    if is_recording:
        return
    is_recording = True
    stop_event.clear()
    try:
        threading.Thread(target=record_and_transcribe, daemon=True).start()
    except Exception:
        is_recording = False
        raise


def stop_recording():
    if not is_recording:
        return
    if not stop_event.is_set():
        stop_event.set()
        status_label.config(text="Status: Stopping...")


# --- BUTTON ---
button_frame = tk.Frame(root, bg="#f8f8f8")
button_frame.pack(pady=20)

start_button = tk.Button(
    button_frame,
    text="Start",
    font=("Arial", 18, "bold"),
    command=start_transcription,
    bg="#4CAF50",
    fg="white",
    width=10,
)
start_button.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=15)

stop_button = tk.Button(
    button_frame,
    text="Stop",
    font=("Arial", 18, "bold"),
    command=stop_recording,
    bg="#D9534F",
    fg="white",
    width=10,
    state=tk.DISABLED,
    disabledforeground="#FFFFFF",
)
stop_button.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=15)

root.mainloop()
