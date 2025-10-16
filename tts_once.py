import sys
import pyttsx3

text = sys.stdin.read().strip()
if not text:
    sys.exit(0)

engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)
engine.say(text)
engine.runAndWait()
engine.stop()
