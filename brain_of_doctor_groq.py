
import base64
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from PIL import Image


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# -----------------------------------
# 2. Encode image for Groq
# -----------------------------------

def encode_image_for_groq(filepath):

    filepath = Path(filepath)

    if not filepath.is_absolute():
        filepath = BASE_DIR / filepath

    if not filepath.exists():
        raise FileNotFoundError(f"Image not found: {filepath}")

    image = Image.open(filepath)
    image.thumbnail((1024, 1024))

    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=75
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# -----------------------------------
# 3. Main Doctor Function
# -----------------------------------

def brain_of_the_doctor(
    patient_text,
    image_filepath=None,
    video_filepath=None
):

    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env file")

    if not image_filepath:
        raise ValueError(
            "Groq vision requires an image. Please upload a skin image."
        )

    # Groq vision processes the image.
    # Video is not directly processed by this implementation.

    image_data = encode_image_for_groq(image_filepath)

    prompt = (
        "You are a confident and natural AI doctor specializing in skin care. "
        "Speak with the reassurance, clarity, and authority of a professional "
        "medical assistant. "
        "Analyze the patient's visible skin condition carefully. "
        "Give general skincare information and cautious recommendations. "
        "Never claim a definite diagnosis based only on an image. "
        "Limit your response to two or three sentences maximum. "
        "Do not use special characters, asterisks, or Markdown formatting "
        "because your response will be converted directly into audio. "
        "If the patient has provided a video, explain that you are reviewing "
        "the uploaded image because this model cannot process video directly.\n\n"
        f"Patient text: {patient_text}"
    )

    if video_filepath:
        prompt += (
            "\nThe patient also uploaded a video, but use the provided "
            "image as the visual reference."
        )

    # -----------------------------------
    # 4. Initialize Groq
    # -----------------------------------

    client = Groq(api_key=groq_api_key)

    # -----------------------------------
    # 5. Send request to Groq
    # -----------------------------------

    response = client.chat.completions.create(
        model=os.environ.get(
            "GROQ_MODEL",
            "qwen/qwen3.8-27b"
        ),
        max_completion_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful skin care assistant. "
                    "Give general information, not a medical diagnosis."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,{image_data}"
                            )
                        }
                    }
                ]
            }
        ]
    )

    doctors_response = response.choices[0].message.content

    if not doctors_response:
        raise ValueError("Groq returned an empty response.")

    return doctors_response
