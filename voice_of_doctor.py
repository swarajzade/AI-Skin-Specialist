
import os
import platform
import subprocess

from pathlib import Path
from dotenv import load_dotenv
from deepgram import DeepgramClient


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_DOCTOR_AUDIO = BASE_DIR / "doctor_response.mp3"


# -----------------------------------
# 2. Convert Text to Doctor Audio
# -----------------------------------

def convert_text_to_doctor_audio(
    text,
    output_filepath=DEFAULT_DOCTOR_AUDIO
):

    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")

    if not deepgram_api_key:
        raise ValueError("Missing DEEPGRAM_API_KEY in .env file")

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    deepgram = DeepgramClient(
        api_key=deepgram_api_key
    )

    # -----------------------------------
    # 3. Generate Audio
    # -----------------------------------

    audio = deepgram.speak.v1.audio.generate(
        text=text,
        model=os.environ.get(
            "DEEPGRAM_TTS_MODEL",
            "aura-2-thalia-en"
        ),
        encoding="mp3"
    )

    # -----------------------------------
    # 4. Save Audio
    # -----------------------------------

    output_filepath = Path(output_filepath)

    if not output_filepath.is_absolute():
        output_filepath = BASE_DIR / output_filepath

    output_filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_filepath.open("wb") as file:

        for chunk in audio:
            file.write(chunk)

    print(f"Doctor audio saved: {output_filepath}")

    return output_filepath


# -----------------------------------
# 5. Play Generated Audio
# -----------------------------------

def play_audio(audio_filepath):

    audio_filepath = Path(audio_filepath)

    if not audio_filepath.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_filepath}"
        )

    audio_filepath = str(audio_filepath.resolve())

    if platform.system() == "Darwin":

        subprocess.run(
            ["afplay", audio_filepath],
            check=False
        )

    elif platform.system() == "Windows":

        os.startfile(audio_filepath)

    else:

        subprocess.run(
            ["xdg-open", audio_filepath],
            check=False
        )
