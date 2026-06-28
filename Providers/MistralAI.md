# Mistral AI SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [docs.mistral.ai](https://docs.mistral.ai)  
**GitHub**: [mistralai/client-python](https://github.com/mistralai/client-python)  
**Package**: `mistralai` (latest)

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Chat Completions](#chat-completions)
- [Available Models](#available-models)
- [Multimodal (Vision)](#multimodal-vision)
- [Tool Use (Function Calling)](#tool-use-function-calling)
- [Embeddings](#embeddings)
- [Error Handling](#error-handling)
- [Structured Outputs](#structured-outputs)
- [Quick Reference](#quick-reference)

---

## Installation & Setup

### Package Installation

```bash
pip install mistralai --upgrade
# Requires Python 3.9+
```

### Client Initialization

```python
from mistralai import Mistral
import os

# Method 1: Auto-discover API key
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# Method 2: Explicit configuration
client = Mistral(
    api_key="your-api-key",
    server_url="https://api.mistral.ai"
)
```

---

## Chat Completions

### `client.chat.complete(model, messages, ...)`

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | `"mistral-large-latest"`, `"mistral-medium-latest"`, `"mistral-small-latest"` |
| `messages` | `list[dict]` | *Required* | Message history with role and content |
| `temperature` | `float` | `0.3` | Randomness (0.0–1.0) |
| `max_tokens` | `int` | `None` | Maximum response length |
| `top_p` | `float` | `1.0` | Nucleus sampling |
| `stop_sequences` | `list[str]` | `None` | Stop generation tokens |

```python
response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "user", "content": "Explain Mistral AI"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Streaming

```python
with client.chat.stream(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Write a poem"}]
) as stream:
    for chunk in stream:
        if chunk.data.choices[0].delta.content:
            print(chunk.data.choices[0].delta.content, end="")
```

---

## Available Models

| Model | Context | Use Case |
|:---|:---|:---|
| `mistral-large-latest` | 128K | General-purpose, reasoning |
| `mistral-medium-latest` | 32K | Balanced speed/quality |
| `mistral-small-latest` | 32K | Fast, cost-efficient |
| `codestral-latest` | 32K | Code generation |
| `pixtral-12b-latest` | 128K | Multimodal (vision) |

---

## Multimodal (Vision)

### Image Input from Base64

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.chat.complete(
    model="pixtral-12b-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image",
                    "image_url": f"data:image/jpeg;base64,{image_data}"
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

---

## Tool Use (Function Calling)

### Tool Definition & Execution

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Args: {tool_call.function.arguments}")
```

---

## Embeddings

### Generate Vector Embeddings

```python
response = client.embeddings.create(
    model="mistral-embed",
    inputs=["Text 1", "Text 2"]
)

for embedding in response.data:
    print(f"Dimension: {len(embedding.embedding)}")
```

---

## Error Handling

### Exception Handling

```python
from mistralai.exceptions import MistralError, APIConnectionError

try:
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Hello"}]
    )
except APIConnectionError as e:
    print(f"Connection error: {e}")
except MistralError as e:
    print(f"API error: {e}")
```

---

## Structured Outputs

Mistral supports enforcing a JSON schema for responses.

```python
response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Extract info: John, 30"}],
    response_format={
        "type": "json_object"
    }
)
```

---

## Quick Reference

```python
from mistralai import Mistral

client = Mistral(api_key="your-key")

# Chat
response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)

# Stream
with client.chat.stream(...) as stream:
    for chunk in stream:
        print(chunk.data.choices[0].delta.content or "", end="")

# Embeddings
embeddings = client.embeddings.create(
    model="mistral-embed",
    inputs=["text"]
)
```

---

**Version**: SDK Latest | **Python Only** | **Updated**: June 2026