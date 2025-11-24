
from openai import OpenAI
import requests, base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

API_KEY = os.getenv("NVIDIA_API_KEY")

headers = {
  "Authorization": f"Bearer {API_KEY}",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "meta/llama-4-maverick-17b-128e-instruct",
  "messages": [{"role":"user","content":"what is machine learning?"}],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 1.00,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": stream
}


response = requests.post(invoke_url, headers=headers, json=payload)

# Handle response
if response.status_code == 200:
    data = response.json()
    message = data["choices"][0]["message"]["content"]
    print("\nAssistant:\n", message)
else:
    print(f"Error {response.status_code}: {response.text}")
