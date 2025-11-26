import tkinter as tk

from tkinter import messagebox, simpledialog

import threading

import json
import io

import os

import numpy as np

import sounddevice as sd

from faster_whisper import WhisperModel
try:
    import requests  # type: ignore
except Exception:
    requests = None
import re
from scipy.io.wavfile import write as wav_write

import pyttsx3

from pathlib import Path
from typing import Optional
import librosa



from command_recognition import CommandRecognizer



BASE_DIR = Path(__file__).parent

COMMAND_DATA_DIR = BASE_DIR / "command_data"

COMMAND_SAMPLE_DIRS = ["samples", os.path.join("Jodis_Recordings", "Recordings")]
COMMANDS_FILE = COMMAND_DATA_DIR / "commands.json"

command_recognizer = None

def reload_command_recognizer():

    global command_recognizer

    try:

        command_recognizer = CommandRecognizer(

            data_dir=COMMAND_DATA_DIR,

            sample_subdirs=COMMAND_SAMPLE_DIRS,

        )

        print("Command recognizer ready.")

        return True

    except Exception as exc:

        command_recognizer = None

        print(f"Command recognizer disabled: {exc}")

        return False

reload_command_recognizer()



# --- SETTINGS ---

LANG = "en"          # or "fr"

MODEL_SIZE = "base"  # tiny, base, small, medium, large

FS = 16000           # Sampling rate

DURATION = 5         # Recording length (seconds)
USE_DEEPGRAM = False  # set False to force local Whisper
PERSONALIZED_MODE = True  # bias output to saved commands only when True
USE_OPENAI_WHISPER = True  # prefer OpenAI Whisper API if key is set
OPENAI_MODEL = "whisper-1"





# --- LOAD WHISPER MODEL ---

print("Loading Whisper model...")

model = WhisperModel(MODEL_SIZE)

print("Model loaded successfully.")



stop_event = threading.Event()

is_recording = False

last_transcript_text = ""
last_audio_capture = None

CORRECTIONS_FILE = os.path.join(os.path.dirname(__file__), "corrections.json")

corrections = {}
save_button = None





def load_corrections():

    global corrections

    if os.path.exists(CORRECTIONS_FILE):

        try:

            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:

                corrections = json.load(f)

        except (json.JSONDecodeError, OSError):

            corrections = {}





def save_corrections():

    try:

        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:

            json.dump(corrections, f, ensure_ascii=False, indent=2)

    except OSError as e:

        messagebox.showerror("Save Error", f"Could not save correction: {e}")





load_corrections()





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
    if USE_OPENAI_WHISPER and requests is not None:
        if not os.getenv("OPENAI_API_KEY"):
            print("OpenAI Whisper skipped: OPENAI_API_KEY not set.")
        else:
            oa_text = transcribe_with_openai(audio)
            if oa_text:
                print("Using OpenAI Whisper API for transcription.")
                return sanitize_transcript(oa_text)
            else:
                print("OpenAI Whisper returned empty; falling back.")
    if USE_DEEPGRAM and requests is not None and os.getenv("DEEPGRAM_API_KEY"):
        dg_text = transcribe_with_deepgram(audio)
        if dg_text:
            return sanitize_transcript(dg_text)

    prompt = ""
    if command_recognizer and command_recognizer.command_names:
        prompt = "Commands: " + ", ".join(command_recognizer.command_names)

    def build_transcript(segments):
        lines = []
        prev_norm = ""
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            norm = text.lower().strip(" ,.")
            # Skip exact or near-duplicate consecutive phrases.
            if norm == prev_norm:
                continue
            prev_norm = norm
            lines.append(text)
        return " ".join(lines)

    segments, _ = model.transcribe(
        audio,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=1.25,
        vad_filter=True,
        language=LANG,
        initial_prompt=prompt,   # <-- bias toward your phrases
    )
    text = build_transcript(segments)
    text = sanitize_transcript(text)
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



content_frame = tk.Frame(root, bg="#f8f8f8")

content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

content_frame.rowconfigure(0, weight=1)

content_frame.rowconfigure(1, minsize=150)

content_frame.columnconfigure(0, weight=1)



text_box = tk.Text(

    content_frame,

    wrap=tk.WORD,

    font=("Arial", 14),

    height=15,

    bg="white"

)

text_box.grid(row=0, column=0, sticky="nsew")



control_frame = tk.Frame(content_frame, bg="#f8f8f8", height=150)

control_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))

control_frame.grid_propagate(False)

control_frame.columnconfigure(0, weight=1)



status_label = tk.Label(

    control_frame,

    text="Status: Idle",

    font=("Arial", 12),

    bg="#f8f8f8"

)

status_label.pack(pady=(10, 5))



buttons_container = tk.Frame(control_frame, bg="#f8f8f8")
buttons_container.pack()

personalized_var = tk.BooleanVar(value=PERSONALIZED_MODE)
personalized_toggle = tk.Checkbutton(
    control_frame,
    text="Personalized (commands only)",
    variable=personalized_var,
    bg="#f8f8f8",
    font=("Arial", 12),
)
personalized_toggle.pack(pady=(0, 5))


# --- TEXT FONT CONTROL ---

def apply_dynamic_font(text_value):

    word_count = len(text_value.split())

    if word_count <= 8:

        font_size = 28

    elif word_count <= 16:

        font_size = 22

    elif word_count <= 32:

        font_size = 18

    else:

        font_size = 14

    text_box.config(font=("Arial", font_size))





def apply_saved_corrections(text_value):
    # Apply any saved manual corrections to the transcribed text.
    if not isinstance(text_value, str):
        return text_value
    key = text_value.strip()
    return corrections.get(key, text_value)


def dedupe_repetitions(text_value: str, max_repeats: int = 1, max_window: int = 12) -> str:
    """Collapse excessive repeated phrases (common Whisper hallucination)."""
    if not isinstance(text_value, str):
        return text_value
    # Split on words but keep commas as separators to avoid merging sentences.
    tokens = text_value.replace(",", " , ").split()
    if len(tokens) < 4:
        return text_value

    result_tokens = []
    i = 0
    while i < len(tokens):
        window_size = min(max_window, len(tokens) - i)
        matched = False
        for w in range(window_size, 1, -1):
            chunk = tokens[i : i + w]
            repeat = 1
            while (
                i + repeat * w + w <= len(tokens)
                and tokens[i + repeat * w : i + (repeat + 1) * w] == chunk
            ):
                repeat += 1
            if repeat > 1:
                # Keep only a limited number of repeats to avoid runaway duplication.
                result_tokens.extend(chunk * min(repeat, max_repeats))
                i += repeat * w
                matched = True
                break
        if not matched:
            result_tokens.append(tokens[i])
            i += 1
    collapsed = " ".join(result_tokens).replace(" ,", ",")
    return " ".join(collapsed.split())


def sanitize_transcript(text_value: str) -> str:
    """Stronger repetition clamp for upstream transcripts."""
    if not isinstance(text_value, str):
        return text_value
    text_value = text_value.strip()
    if not text_value:
        return text_value

    # Remove immediate single-word runs (word word word -> word).
    tokens = text_value.replace(",", " , ").split()
    collapsed_tokens = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        j = i + 1
        while j < len(tokens) and tokens[j] == t:
            j += 1
        collapsed_tokens.append(t)
        i = j
    collapsed = " ".join(collapsed_tokens).replace(" ,", ",")

    # Deduplicate clauses split by punctuation; keep first occurrence.
    clauses = [c.strip() for c in re.split(r"[.;,]+", collapsed) if c.strip()]
    seen = set()
    unique_clauses = []
    for c in clauses:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_clauses.append(c)
    return ", ".join(unique_clauses)


def _to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 mono audio (-1..1) to 16-bit PCM bytes."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    return ints.tobytes()


def transcribe_with_deepgram(audio: np.ndarray) -> str:
    """Send audio to Deepgram; return transcript or '' on failure."""
    if requests is None:
        return ""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        return ""
    keywords = []
    if command_recognizer and command_recognizer.command_names:
        keywords = list(command_recognizer.command_names)
    try:
        pcm = _to_pcm16(audio)
        if not pcm:
            return ""
        resp = requests.post(
            "https://api.deepgram.com/v1/listen",
            params={
                "model": "nova-2-general",
                "language": LANG,
                "encoding": "linear16",
                "sample_rate": FS,
                "channels": 1,
                "punctuate": "true",
                "smart_format": "true",
                "paragraphs": "false",
                "utterances": "false",
                "no_delay": "true",
                "diarize": "false",
                # Keyword boosting to favor your commands (mild boost).
                "keywords": ",".join(keywords) if keywords else None,
            },
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/octet-stream",
            },
            data=pcm,
            timeout=20,
        )
        if not resp.ok:
            print(f"Deepgram failed: {resp.status_code} {resp.text}")
            return ""
        data = resp.json()
        alts = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [])
        if not alts:
            return ""
        return alts[0].get("transcript", "") or ""
    except Exception as exc:
        print(f"Deepgram failed: {exc}")
        return ""


def transcribe_with_openai(audio: np.ndarray) -> str:
    """Send audio to OpenAI Whisper API; return transcript or '' on failure."""
    if requests is None:
        return ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    prompt = ""
    try:
        if command_recognizer and command_recognizer.command_names:
            prompt = "Commands: " + ", ".join(command_recognizer.command_names)
    except Exception:
        prompt = ""
    try:
        pcm = np.asarray(audio, dtype=np.float32)
        if pcm.size == 0:
            return ""
        buf = io.BytesIO()
        wav_write(buf, FS, (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16))
        buf.seek(0)
        files = {"file": ("audio.wav", buf, "audio/wav")}
        data = {"model": OPENAI_MODEL, "language": LANG}
        if prompt:
            data["prompt"] = prompt
        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files=files,
            timeout=30,
        )
        if not resp.ok:
            print(f"OpenAI Whisper failed: {resp.status_code} {resp.text}")
            return ""
        out = resp.json()
        return out.get("text", "") or ""
    except Exception as exc:
        print(f"OpenAI Whisper failed: {exc}")
        return ""


def extract_training_features(audio: np.ndarray, sr: int = FS):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("No audio captured to save.")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return combined.T.astype(float).tolist()


def load_commands_data():
    if COMMANDS_FILE.exists():
        try:
            with COMMANDS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_commands_data(data):
    COMMANDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with COMMANDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_current_audio_as_command():
    global last_audio_capture
    if last_audio_capture is None or last_audio_capture.size == 0:
        messagebox.showinfo("No audio", "Record something before saving a command sample.")
        return

    default_label = last_transcript_text.strip() or ""
    label = simpledialog.askstring(
        "Save command",
        "Enter the command phrase to associate with this recording:",
        initialvalue=default_label,
        parent=root,
    )
    if label is None:
        return
    label = label.strip()
    if not label:
        messagebox.showinfo("Empty label", "Please provide a command name.")
        return

    try:
        features = extract_training_features(last_audio_capture)
    except ValueError as exc:
        messagebox.showerror("Unable to save", str(exc))
        return

    data = load_commands_data()
    entry = data.setdefault(label, {})
    entry.setdefault("samples", [])
    entry["samples"].append(features)

    save_commands_data(data)
    messagebox.showinfo("Saved", f"Recording saved under '{label}'.")
    if save_button is not None:
        save_button.config(state=tk.DISABLED)
    if reload_command_recognizer():
        status_label.config(text="Status: Commands reloaded")





def summarize_command_matches(matches):

    if not matches:

        return None



    lines = []

    transcript_hit = matches.get("transcript")

    if transcript_hit:

        lines.append(

            f"Transcript keyword match: {transcript_hit['command']} (score {transcript_hit['score']:.2f})"

        )

    text_fuzzy = matches.get("text_fuzzy")
    if text_fuzzy:

        lines.append(

            f"Fuzzy transcript match: {text_fuzzy['command']} (confidence {text_fuzzy['score']:.2f})"

        )



    dtw_hit = matches.get("dtw")

    if dtw_hit:

        lines.append(

            f"DTW template match: {dtw_hit['command']} (confidence {dtw_hit['score']:.2f})"

        )



    yamnet_hit = matches.get("yamnet")

    if yamnet_hit:

        dist = yamnet_hit.get("distance")

        score = yamnet_hit.get("score")

        if score is not None and dist is not None:

            lines.append(

                f"YAMNet similarity: {yamnet_hit['command']} (score {score:.2f}, distance {dist:.2f})"

            )

        else:

            lines.append(f"YAMNet similarity: {yamnet_hit['command']}")



    if not lines:

        return None



    return "Command hints:\n" + "\n".join(f"- {line}" for line in lines)


def select_best_command(matches: Optional[dict]) -> Optional[str]:
    """Pick a single best command from transcript/DTW/YAMNet."""
    if not matches:
        return None
    # Priority: DTW > transcript > YAMNet with confidence gating.
    if matches.get("dtw"):
        score = matches["dtw"].get("score", 0.0)
        margin = matches["dtw"].get("margin", 0.0)
        if score >= 0.65 and margin >= 0.06:
            return matches["dtw"]["command"]
    if matches.get("transcript"):
        return matches["transcript"]["command"]
    if matches.get("text_fuzzy") and matches["text_fuzzy"].get("score", 0.0) >= 0.55:
        return matches["text_fuzzy"]["command"]
    if matches.get("yamnet"):
        return matches["yamnet"]["command"]
    return None





# --- MAIN FUNCTION ---

def record_and_transcribe():

    global is_recording, last_transcript_text, last_audio_capture

    is_recording = True

    status_label.config(text="Status: Recording...")

    text_box.delete(1.0, tk.END)

    start_button.config(state=tk.DISABLED)

    stop_button.config(state=tk.NORMAL)

    not_button.config(state=tk.DISABLED)

    last_transcript_text = ""



    try:

        audio = record_audio(stop_event)

        last_audio_capture = audio.copy()

        stop_button.config(state=tk.DISABLED)

        was_stopped = stop_event.is_set()

        stop_event.clear()



        if audio.size == 0:

            status_text = "Status: Recording stopped" if was_stopped else "Status: No audio captured"

            status_label.config(text=status_text)

            return



        status_label.config(text="Status: Transcribing...")

        result_text = transcribe_audio(audio)

        result_text = apply_saved_corrections(result_text)
        result_text = dedupe_repetitions(result_text)



        matches = None
        summary = None
        if command_recognizer is not None:
            try:
                matches = command_recognizer.match(audio, result_text)
            except Exception as exc:
                print(f"Command recognizer error: {exc}")
                matches = None
            summary = summarize_command_matches(matches)

        if personalized_var.get():
            best_cmd = select_best_command(matches)
            if best_cmd:
                result_text = best_cmd
            else:
                result_text = ""

        text_box.delete(1.0, tk.END)
        if result_text:
            apply_dynamic_font(result_text)
            text_box.insert(tk.END, f"You said:\n{result_text}\n\n")
            if summary:
                text_box.insert(tk.END, f"{summary}\n\n")
            text_box.update_idletasks()

            status_label.config(text="Status: Speaking...")
            speak_text(result_text)
            last_transcript_text = result_text
        else:
            placeholder = "[No command detected]" if personalized_var.get() else "[No transcription]"
            text_box.insert(tk.END, f"You said:\n{placeholder}\n\n")
            last_transcript_text = ""


        status_label.config(text="Status: Done")
        if save_button is not None:
            if last_audio_capture is not None and last_audio_capture.size > 0:
                save_button.config(state=tk.NORMAL)
            else:
                save_button.config(state=tk.DISABLED)

    finally:

        start_button.config(state=tk.NORMAL)

        stop_button.config(state=tk.DISABLED)

        not_button.config(state=tk.NORMAL if last_transcript_text.strip() else tk.DISABLED)

        is_recording = False





def start_transcription():

    global is_recording

    if is_recording:

        return

    is_recording = True

    stop_event.clear()

    if 'save_button' in globals():

        try:

            save_button.config(state=tk.DISABLED)

        except Exception:

            pass

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





def mark_incorrect_transcription():

    global corrections, last_transcript_text, last_audio_capture

    current_text = last_transcript_text.strip()

    if not current_text:

        messagebox.showinfo("No transcription", "No recent transcription is available to correct.")

        return



    correction = simpledialog.askstring(

        "Not what I said",

        "Enter what was actually said:",

        parent=root,

    )

    if correction is None:

        return

    correction = correction.strip()

    if not correction:

        messagebox.showinfo("Empty correction", "Please provide the word or sentence that was actually said.")

        return



    corrections[current_text] = correction

    save_corrections()

    last_transcript_text = correction

    apply_dynamic_font(correction)

    text_box.delete(1.0, tk.END)

    text_box.insert(tk.END, correction)

    status = "Status: Correction saved."

    # Also save the current audio as a DTW template under the corrected label.
    audio_saved = False
    if last_audio_capture is not None and getattr(last_audio_capture, "size", 0) > 0:
        try:
            features = extract_training_features(last_audio_capture)
            data = load_commands_data()
            entry = data.setdefault(correction, {})
            entry.setdefault("samples", [])
            entry["samples"].append(features)
            save_commands_data(data)
            audio_saved = True
            status = f"Status: Correction saved; sample added to '{correction}'."
            if save_button is not None:
                save_button.config(state=tk.DISABLED)
            reload_command_recognizer()
        except Exception as exc:
            print(f"Failed to save corrected sample: {exc}")

    status_label.config(text=status)

    not_button.config(state=tk.NORMAL)





# --- BUTTON ---

start_button = tk.Button(

    buttons_container,

    text="Start",

    font=("Arial", 18, "bold"),

    command=start_transcription,

    bg="#4CAF50",

    fg="white",

    width=10,

    height=2,

)

start_button.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=15)



stop_button = tk.Button(

    buttons_container,

    text="Stop",

    font=("Arial", 18, "bold"),

    command=stop_recording,

    bg="#D9534F",

    fg="white",

    width=10,

    height=2,

    state=tk.DISABLED,

    disabledforeground="#FFFFFF",

)

stop_button.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=15)



not_button = tk.Button(

    buttons_container,

    text="Not what I said",

    font=("Arial", 18, "bold"),

    command=mark_incorrect_transcription,

    bg="#F0AD4E",

    fg="white",

    width=12,

    height=2,

    state=tk.DISABLED,

    disabledforeground="#FFFFFF",

)

not_button.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=15)

save_button = tk.Button(
    buttons_container,
    text="Save Command",
    font=("Arial", 16, "bold"),
    command=save_current_audio_as_command,
    bg="#5BC0DE",
    fg="white",
    width=14,
    height=2,
    state=tk.DISABLED,
    disabledforeground="#FFFFFF",
)
save_button.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=15)




root.mainloop()


