import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from groq import Groq

folder = os.path.dirname(__file__)
env_path = os.path.join(folder, ".env")
load_dotenv(env_path)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

image_path = os.path.join(folder, "image.jpg")
with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")
client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b "), 
    max_tokens=1000,
    messages=[
            {
                "role": "system",
                "content": "You are a careful skin care assistant. Give general information, not a diagnosis.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image? Give general information, not a diagnosis."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        },
                    },
                ],
            },
        ],
    )

print(response.choices[0].message.content) 