import sounddevice as sd
from scipy.io.wavfile import write

# 🎛️ Paramètres d'enregistrement
FS = 16000      # fréquence d’échantillonnage (Hz)
DURATION = 5    # durée de l’enregistrement (secondes)
FILENAME = "test.wav"

print("Recording... Speak now in English!")
audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
sd.wait()

write(FILENAME, FS, audio)
print(f"Recording finished. File saved as: {FILENAME}")
