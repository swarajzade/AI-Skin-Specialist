
import os
import logging

import speech_recognition as sr

from pathlib import Path
from io import BytesIO
from pydub import AudioSegment
from dotenv import load_dotenv
from groq import Groq


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_PATIENT_AUDIO = BASE_DIR / "patient_voice.mp3"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------------
# 2. Record Patient Audio
# -----------------------------------

def record_audio(
    file_path=DEFAULT_PATIENT_AUDIO,
    timeout=20,
    phrase_time_limit=10
):

    recognizer = sr.Recognizer()

    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with sr.Microphone() as source:

        logging.info("Adjusting for ambient noise...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

        logging.info("Start speaking now...")

        audio_data = recognizer.listen(
            source,
            timeout=timeout,
            phrase_time_limit=phrase_time_limit
        )

        logging.info("Recording complete.")

        # Convert recorded audio to WAV
        wav_data = audio_data.get_wav_data()

        # Convert WAV to MP3
        audio_segment = AudioSegment.from_wav(
            BytesIO(wav_data)
        )

        audio_segment.export(
            str(file_path),
            format="mp3",
            bitrate="128k"
        )

        logging.info(f"Audio saved to {file_path}")

    return file_path


# -----------------------------------
# 3. Convert Patient Voice to Text
# -----------------------------------

def transcribe_patient_voice(
    audio_filepath=None,
    timeout=20,
    phrase_time_limit=10
):

    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env file")

    # If no audio file is provided, record patient voice
    if audio_filepath is None:

        audio_filepath = record_audio(
            file_path=DEFAULT_PATIENT_AUDIO,
            timeout=timeout,
            phrase_time_limit=phrase_time_limit
        )

    else:

        audio_filepath = Path(audio_filepath)

        if not audio_filepath.is_absolute():
            audio_filepath = BASE_DIR / audio_filepath

    if not audio_filepath.exists():
        raise FileNotFoundError(
            f"Patient audio not found: {audio_filepath}"
        )

    # -----------------------------------
    # 4. Initialize Groq
    # -----------------------------------

    client = Groq(
        api_key=groq_api_key
    )

    # -----------------------------------
    # 5. Transcribe Audio
    # -----------------------------------

    logging.info("Transcribing patient voice...")

    with audio_filepath.open("rb") as audio_file:

        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=os.environ.get(
                "WHISPER_MODEL",
                "whisper-large-v3"
            )
        )

    patient_text = transcription.text

    if not patient_text:
        raise ValueError("Could not transcribe patient audio.")

    logging.info("Patient voice transcription completed.")

    print("\nPATIENT SAID:")
    print(patient_text)

    return patient_text
