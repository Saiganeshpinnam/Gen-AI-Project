from gtts import gTTS
import os
import uuid

def text_to_speech(text: str) -> str:
    """
    Convert text to speech and return audio file path
    """
    if not text.strip():
        return None

    # Unique filename
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("audio", filename)

    # Create audio folder if not exists
    os.makedirs("audio", exist_ok=True)

    tts = gTTS(text=text, lang="en")
    tts.save(filepath)

    return filepath
