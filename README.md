# Voice Transcription & Speech Feedback App

This project is a **real-time voice transcription and speech feedback system** built with **Python** and **Whisper (OpenAI)**.  
It allows users to speak into their microphone, see the **live transcription** on screen, and hear their own words **spoken back by the computer**.

---

## Features
- Live voice capture using your computer’s microphone  
- Real-time transcription using **Whisper** 
- Addapted for usage with a computer without any GPU or RAM 
- Optional **text-to-speech** output — your computer repeats what you said  
- Simple and clean **Tkinter GUI**  
- Works offline (no internet required after installation)

---

## Requirements
- **Python 3.10+** (recommended 3.11 or newer)
- A working **microphone**
- An environment that supports **Tkinter** (included with most Python installations)

---

## Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/Dally711/voice-transcription-project.git
cd voice-transcription-project
```

---

### 2. Create a virtual environment
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies
You can install everything with one command:
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, manually install the key libraries:
```bash
pip install faster-whisper sounddevice numpy pyttsx3 tkinter
```

---

### 4. Run the program
Make sure your virtual environment is active, then run:
```bash
python live_gui_transcribe.py
```

---

## How It Works
1. The app records your microphone input in short intervals.  
2. Each audio chunk is transcribed in real-time using **Whisper**.  
3. The recognized text is displayed in the GUI.  
4. A background speech engine converts the text back to speech, allowing the computer to say what you said.  

---

## Project Structure
```
voice-transcription-project/
│
├── live_gui_transcribe.py   # Main GUI and real-time transcription logic
├── speech_output.py         # Handles text-to-speech output
├── record_wav.py            # (Optional) record short test clips
├── venv/                    # Virtual environment (excluded in .gitignore)
├── requirements.txt         # List of required dependencies
└── README.md                # You are here
```

---

## Tips
- For faster transcription, use the **tiny** or **base** Whisper model instead of **medium** or **large**.  
- For higher accuracy, use **medium** or **large** if your system is powerful enough.  
- If you have an **NVIDIA GPU**, install **CUDA** — Whisper will automatically use it for a huge speed boost.

---

## Credits
This project is based on the open-source [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) repository, modified and extended by **Jaïme Tapa** to include:
- A graphical interface  
- Real-time speech feedback  
- Simplified setup and offline usability

---

## Author
**Jaïme D. Tapa**  
Software Engineering Student @ University of Ottawa  
GitHub: [Dally711](https://github.com/Dally711)
