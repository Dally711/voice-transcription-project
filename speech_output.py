import pyttsx3
import sys

engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

print("Speech engine ready.")

for line in sys.stdin:
    text = line.strip()
    if text.lower() == "exit":
        break
    if text:
        engine.say(text)
        engine.runAndWait()

engine.stop()
