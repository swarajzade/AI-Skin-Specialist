
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import cv2
import mimetypes


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


# -----------------------------------
# 2. Initialize OpenRouter
# -----------------------------------

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3.5-super-vl:free"
)


# -----------------------------------
# 3. Helper functions
# -----------------------------------

def encode_file(filepath):

    with open(filepath, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_media_type(filepath, fallback):

    media_type, _ = mimetypes.guess_type(filepath)

    return media_type or fallback


# -----------------------------------
# 4. Main Doctor Function
# -----------------------------------

def brain_of_the_doctor(
    patient_text,
    image_filepath=None,
    video_filepath=None
):

    folder = os.path.dirname(os.path.abspath(__file__))

    # Resolve file paths
    if image_filepath and not os.path.isabs(image_filepath):
        image_filepath = os.path.join(folder, image_filepath)

    if video_filepath and not os.path.isabs(video_filepath):
        video_filepath = os.path.join(folder, video_filepath)

    if not image_filepath and not video_filepath:
        raise ValueError("Please provide an image or video.")

    if image_filepath and video_filepath:
        raise ValueError("Provide either an image or a video, not both.")

    # -----------------------------------
    # 5. Doctor System Prompt
    # -----------------------------------

    prompt = (
        "You are a confident, natural AI doctor specializing in skin care. "
        "Speak with the reassurance, clarity, and authority of a professional "
        "medical assistant. "
        "Analyze the patient's visible skin condition carefully. "
        "Describe relevant visible symptoms and provide cautious guidance. "
        "Do not claim a definite diagnosis from an image or video. "
        "If the patient provides only an image and no video, your absolute "
        "priority is to ask them to provide a video showing the skin problem "
        "so you can understand it better. "
        "Limit your entire response to two or three sentences maximum. "
        "Do not use special characters, symbols, asterisks, or Markdown "
        "formatting because the response will be converted directly to audio.\n\n"
        f"Patient text: {patient_text}"
    )

    # -----------------------------------
    # 6. Prepare Message Content
    # -----------------------------------

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    # -----------------------------------
    # 7. Process Video
    # -----------------------------------

    if video_filepath:

        if not os.path.exists(video_filepath):
            raise FileNotFoundError(
                f"Video not found: {video_filepath}"
            )

        cap = cv2.VideoCapture(video_filepath)

        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            cap.release()
            raise ValueError("Could not read video FPS.")

        frame_interval = max(1, int(fps * 2))

        frame_count = 0
        frames_processed = 0

        print("Extracting video frames...")

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                if frame_count % frame_interval == 0:

                    success, buffer = cv2.imencode(".jpg", frame)

                    if success:

                        frame_data = base64.b64encode(
                            buffer
                        ).decode("utf-8")

                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:image/jpeg;base64,{frame_data}"
                                    )
                                }
                            }
                        )

                        frames_processed += 1

                        print(
                            f"Extracted frame: {frames_processed}"
                        )

                    if frames_processed >= 10:
                        break

                frame_count += 1

        finally:
            cap.release()

        if frames_processed == 0:
            raise ValueError("No frames extracted from video.")

    # -----------------------------------
    # 8. Process Image
    # -----------------------------------

    elif image_filepath:

        if not os.path.exists(image_filepath):
            raise FileNotFoundError(
                f"Image not found: {image_filepath}"
            )

        image_data = encode_file(image_filepath)

        media_type = get_media_type(
            image_filepath,
            "image/jpeg"
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{media_type};base64,{image_data}"
                    )
                }
            }
        )

    # -----------------------------------
    # 9. Send Request to OpenRouter
    # -----------------------------------

    print("\nAnalyzing patient's skin condition...\n")

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    # -----------------------------------
    # 10. Extract Doctor Response
    # -----------------------------------

    doctors_response_in_text = (
        response.choices[0].message.content
    )

    if not doctors_response_in_text:
        raise ValueError("AI returned an empty response.")

    return doctors_response_in_text
