# Groq SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [console.groq.com/docs](https://console.groq.com/docs)  
**GitHub**: [groq/groq-python](https://github.com/groq/groq-python)  
**Package**: `groq` (latest)

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Chat Completions (OpenAI-compatible)](#chat-completions-openai-compatible)
- [Available Models](#available-models)
- [Multimodal (Vision)](#multimodal-vision)
- [Tool Use (Function Calling)](#tool-use-function-calling)
- [Audio (Whisper)](#audio-whisper)
- [Error Handling](#error-handling)
- [Whisper Large V3 Turbo](#whisper-large-v3-turbo)
- [Quick Reference](#quick-reference)

---

## Installation & Setup

### Package Installation

```bash
pip install groq --upgrade
# Requires Python 3.7+
```

### Client Initialization

```python
from groq import Groq
import os

# Method 1: Auto-discover API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Method 2: Explicit configuration
client = Groq(
    api_key="your-api-key",
    timeout=30.0,
    max_retries=2
)

# Async client
from groq import AsyncGroq
async_client = AsyncGroq(api_key="your-api-key")
```

---

## Chat Completions (OpenAI-compatible)

### `client.chat.completions.create(model, messages, ...)`

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | `"llama-3.3-70b-versatile"`, `"llama-3.1-70b-versatile"` |
| `messages` | `list[dict]` | *Required* | Message history |
| `temperature` | `float` | `0.7` | Randomness (0.0–2.0) |
| `max_tokens` | `int` | `1024` | Maximum response length |
| `top_p` | `float` | `1.0` | Nucleus sampling |
| `stream` | `bool` | `False` | Enable streaming |

```python
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Explain the importance of fast LLMs"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(completion.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Available Models

| Model | Context | Tokens/Sec | Best For |
|:---|:---|:---:|:---|
| `llama-3.3-70b-versatile` | 8K | ~300 | General-purpose |
| `llama-3.1-70b-versatile` | 128K | ~250 | Long context |
| `llama-3.1-8b-instant` | 128K | ~400 | Fast, lightweight |
| `mixtral-8x7b-32768` | 32K | ~250 | Expert mixture |

---

## Multimodal (Vision)

### Image Input

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    ]
)

print(completion.choices[0].message.content)
```

---

## Tool Use (Function Calling)

### Define and Execute Tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

completion = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=tools,
    tool_choice="auto"
)

if completion.choices[0].message.tool_calls:
    for tool_call in completion.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Args: {tool_call.function.arguments}")
```

---

## Audio (Whisper)

### Speech-to-Text Transcription

```python
with open("audio.m4a", "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=("audio.m4a", f),
        model="whisper-large-v3",
        language="en"
    )

print(f"Transcription: {transcription.text}")
```

### Text-to-Speech

```python
response = client.audio.speech.create(
    model="playai-tts",
    voice="Fritz-PlayAI",
    input="Hello from Groq!",
    response_format="wav"
)

with open("output.wav", "wb") as f:
    f.write(response.read())
```

---

## Error Handling

### Exception Handling

```python
from groq import Groq
from groq.exceptions import RateLimitError, APIConnectionError

try:
    completion = client.chat.completions.create(...)
except RateLimitError:
    print("Rate limited. Implement backoff.")
except APIConnectionError:
    print("Connection error. Retry.")
```

---

## Whisper Large V3 Turbo

Groq provides extremely fast audio transcription using the Whisper Large V3 Turbo model.

```python
with open("audio.m4a", "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=("audio.m4a", f),
        model="whisper-large-v3-turbo",
        language="en"
    )
print(transcription.text)
```

---

## Quick Reference

```python
from groq import Groq

client = Groq(api_key="your-key")

# Chat
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello"}]
)
print(completion.choices[0].message.content)

# Stream
stream = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[...],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# Transcribe
transcription = client.audio.transcriptions.create(
    file=open("audio.m4a", "rb"),
    model="whisper-large-v3"
)
```

---

**Version**: SDK Latest | **Python Only** | **Updated**: June 2026