from gtts import gTTS

# Create alerts with different messages and urgency levels
tts = gTTS("Warning! Vehicle detected!", lang='en')
tts.save("vehicle_alert.mp3")

tts = gTTS("Caution! Person in path!", lang='en')
tts.save("person_alert.mp3")

tts = gTTS("Alert! Animal detected!", lang='en')
tts.save("animal_alert.mp3")

tts = gTTS("Warning! Barrier ahead!", lang='en')
tts.save("barrier_alert.mp3")

tts = gTTS("Caution!", lang='en')
tts.save("alert.mp3")
