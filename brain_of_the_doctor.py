
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import cv2

# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


# -----------------------------------
# 2. Initialize OpenRouter
# -----------------------------------

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

folder = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------
# 3. Load video
# -----------------------------------

video_path = os.path.join(folder, "video.mp4")

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Could not open video file")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise ValueError("Could not read video FPS")


# -----------------------------------
# 4. Extract video frames
# -----------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Analyze this video carefully. "
                    "Describe what is happening, the visible "
                    "details, and any important changes "
                    "between the frames."
                )
            }
        ]
    }
]

frame_interval = max(1, int(fps * 2))

frame_count = 0
frames_processed = 0

print("Extracting video frames...")

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

            messages[0]["content"].append(
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

            print(f"Extracted frame: {frames_processed}")

        if frames_processed >= 10:
            break

    frame_count += 1

cap.release()

if frames_processed == 0:
    raise ValueError("No frames extracted from video")


# -----------------------------------
# 5. Send frames to AI model
# -----------------------------------

print("\nAnalyzing video with AI...\n")

response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-super-vl:free",
    max_tokens=1000,
    messages=messages
)


# -----------------------------------
# 6. Display response
# -----------------------------------

print("\nAI DOCTOR RESPONSE:\n")

print(response.choices[0].message.content)

