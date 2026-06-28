# Google Gemini SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [ai.google.dev](https://ai.google.dev/gemini-api/docs)  
**GitHub**: [googleapis/python-genai](https://github.com/googleapis/python-genai)  
**Package**: `google-genai` (latest)

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Chat API](#chat-api)
- [Multimodal Input](#multimodal-input)
- [Tool Use (Function Calling)](#tool-use-function-calling)
- [Structured Outputs (JSON)](#structured-outputs-json)
- [Embeddings](#embeddings)
- [Code Execution](#code-execution)
- [Quick Reference](#quick-reference)

---

## Installation & Setup

### Package Installation

```bash
pip install google-genai --upgrade
# Requires Python 3.9+
```

### Client Initialization

```python
from google import genai
import os

# Method 1: Auto-discover API key
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Method 2: Vertex AI (GCP)
client = genai.Client(
    vertexai=True,
    project="your-project-id",
    location="us-central1"
)
```

---

## Chat API

### `client.models.generate_content(model, contents, ...)`

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | `"gemini-2.5-flash"`, `"gemini-2.5-pro"` |
| `contents` | `str \| list` | *Required* | Text prompt or message list |
| `temperature` | `float` | `1.0` | Randomness (0.0–2.0) |
| `max_output_tokens` | `int` | `None` | Maximum response length |
| `stream` | `bool` | `False` | Enable streaming |

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain quantum computing"
)

print(response.text)
```

### Multi-turn Conversation

```python
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=genai.types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=1024
    )
)

# Turn 1
response1 = chat.send_message("What is AI?")
print(f"Assistant: {response1.text}\n")

# Turn 2 — context preserved
response2 = chat.send_message("Can you explain neural networks?")
print(f"Assistant: {response2.text}")
```

### Streaming

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a story",
    stream=True
)

for chunk in response:
    if chunk.text:
        print(chunk.text, end="")
```

---

## Multimodal Input

### Image Processing

```python
from pathlib import Path

image_part = genai.upload_file(path="photo.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "Analyze this image:",
        image_part
    ]
)

print(response.text)
```

### PDF/Document Upload

```python
pdf_file = genai.upload_file(
    path="large_document.pdf",
    mime_type="application/pdf"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "Summarize this document in 3 bullet points",
        pdf_file
    ]
)

print(response.text)
```

### Video Upload

```python
video_file = genai.upload_file(
    path="sample.mp4",
    mime_type="video/mp4"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "Describe what happens in this video",
        video_file
    ]
)
```

---

## Tool Use (Function Calling)

### Define and Execute Tools

```python
from google import genai

tools = [
    genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name="get_weather",
                description="Get weather for a location",
                parameters=genai.types.Schema(
                    type="OBJECT",
                    properties={
                        "location": genai.types.Schema(type="STRING"),
                        "unit": genai.types.Schema(
                            type="STRING",
                            enum=["celsius", "fahrenheit"]
                        )
                    },
                    required=["location"]
                )
            )
        ]
    )
]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Weather in Paris?",
    tools=tools
)

if response.tool_calls:
    for tool_call in response.tool_calls[0].function_calls:
        print(f"Called: {tool_call.name}")
        print(f"Args: {tool_call.args}")
```

---

## Structured Outputs (JSON)

### JSON Schema Response

```python
from google import genai

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Extract: John Doe, 28, Engineer",
    config=genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type="OBJECT",
            properties={
                "name": genai.types.Schema(type="STRING"),
                "age": genai.types.Schema(type="INTEGER")
            },
            required=["name", "age"]
        )
    )
)

import json
data = json.loads(response.text)
print(f"Name: {data['name']}")
```

---

## Embeddings

### Generate Vector Embeddings

```python
response = client.models.embed_content(
    model="text-embedding-004",
    content="This is a sample sentence."
)

embedding = response.embedding
print(f"Dimension: {len(embedding)}")

# Batch embeddings
response = client.models.embed_content(
    model="text-embedding-004",
    content=["Text 1", "Text 2", "Text 3"]
)

for i, emb in enumerate(response.embeddings):
    print(f"Text {i}: {len(emb)} dims")
```

---

## Code Execution

Gemini can execute Python code in a sandboxed environment to solve complex reasoning and math problems.

### Enable Code Execution

```python
from google import genai

client = genai.Client(api_key="your-api-key")
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Calculate the 100th Fibonacci number.",
    tools=[genai.types.Tool(code_execution=genai.types.CodeExecution())]
)

print(response.text)
```

---

## Quick Reference

```python
from google import genai

client = genai.Client(api_key="your-key")

# Simple generation
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello"
)
print(response.text)

# Streaming
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Tell a story",
    stream=True
)
for chunk in response:
    print(chunk.text, end="")

# Embeddings
embeddings = client.models.embed_content(
    model="text-embedding-004",
    content=["text"]
)
```

---

**Version**: SDK Latest | **Python Only** | **Updated**: June 2026