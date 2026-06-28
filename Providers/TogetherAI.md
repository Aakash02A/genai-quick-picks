# Together AI SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [docs.together.ai](https://docs.together.ai)  
**GitHub**: [togethercomputer/together-python](https://github.com/togethercomputer/together-python)  
**Package**: `together` (latest)

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Chat Completions](#chat-completions)
- [Available Models](#available-models)
- [Image Generation](#image-generation)
- [Embeddings](#embeddings)
- [Fine-tuning](#fine-tuning)
- [Error Handling](#error-handling)
- [Meta Llama 3.1 405B](#meta-llama-31-405b)
- [Quick Reference](#quick-reference)

---

## Installation & Setup

### Package Installation

```bash
pip install together --upgrade
# Requires Python 3.10+
```

### Client Initialization

```python
from together import Together
import os

# Method 1: Auto-discover API key
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Method 2: Explicit configuration
client = Together(
    api_key="your-api-key",
    timeout=30.0
)

# Async client
from together import AsyncTogether
async_client = AsyncTogether(api_key="your-api-key")
```

---

## Chat Completions

### `client.chat.completions.create(model, messages, ...)`

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | `"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"` |
| `messages` | `list[dict]` | *Required* | Message history |
| `temperature` | `float` | `0.7` | Randomness (0.0–2.0) |
| `max_tokens` | `int` | `512` | Maximum response length |
| `top_p` | `float` | `1.0` | Nucleus sampling |
| `top_k` | `int` | `None` | Top-K sampling |
| `stream` | `bool` | `False` | Enable streaming |

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[
        {"role": "user", "content": "Explain distributed systems"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Write code"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Async Batch

```python
import asyncio

async def batch_requests():
    client = AsyncTogether()
    
    tasks = [
        client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        for prompt in ["What is AI?", "Explain ML", "Define DL"]
    ]
    
    responses = await asyncio.gather(*tasks)
    for i, resp in enumerate(responses):
        print(f"Response {i}: {resp.choices[0].message.content}\n")

asyncio.run(batch_requests())
```

---

## Available Models

| Model | Context | Type |
|:---|:---|:---|
| `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | 128K | General-purpose |
| `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | 128K | Lightweight |
| `mistralai/Mistral-7B-Instruct-v0.2` | 32K | Efficient |
| `meta-llama/CodeLlama-70b-Instruct-hf` | 100K | Code-focused |

---

## Image Generation

### Generate Images

```python
response = client.images.generate(
    prompt="A serene mountain landscape at sunset",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    width=1024,
    height=1024,
    steps=20,
    seed=42
)

for i, img in enumerate(response.data):
    # Save or process
    print(f"Image {i} generated")
```

**Image Models**

| Model | Quality | Speed |
|:---|:---|:---|
| `stabilityai/stable-diffusion-xl-base-1.0` | High | Medium |
| `black-forest-labs/FLUX.1-dev` | Very High | Slower |
| `black-forest-labs/FLUX.1-schnell` | High | Fast |

---

## Embeddings

### Generate Vector Embeddings

```python
response = client.embeddings.create(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    input=["Text 1", "Text 2"]
)

for i, embedding in enumerate(response.data):
    print(f"Text {i}: {len(embedding.embedding)} dimensions")
```

---

## Fine-tuning

### Train Custom Model

```python
import time

# Upload training data
with open("training_data.jsonl", "rb") as f:
    uploaded_file = client.files.upload(file=f)

# Create job
job = client.fine_tuning.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    training_file=uploaded_file.id,
    n_epochs=3,
    learning_rate=0.001,
    batch_size=8
)

print(f"Job: {job.id}")

# Monitor
while True:
    job_status = client.fine_tuning.get(job.id)
    print(f"Status: {job_status.status}")
    
    if job_status.status in ["COMPLETED", "FAILED"]:
        break
    
    time.sleep(10)

# Use fine-tuned model
if job_status.status == "COMPLETED":
    response = client.chat.completions.create(
        model=job_status.id,
        messages=[{"role": "user", "content": "Hello"}]
    )
```

---

## Error Handling

### Exception Handling

```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    if e.status == 429:
        print("Rate limited. Implement backoff.")
    elif e.status >= 500:
        print("Server error. Retry.")
    else:
        print(f"Error: {e.message}")
```

---

## Meta Llama 3.1 405B

Together AI supports serving the massive 405B parameter version of Llama 3.1.

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    max_tokens=512
)
print(response.choices[0].message.content)
```

---

## Quick Reference

```python
from together import Together

client = Together(api_key="your-key")

# Chat
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)

# Stream
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[...],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# Embeddings
embeddings = client.embeddings.create(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    input=["text1", "text2"]
)

# Image generation
image = client.images.generate(
    prompt="A beautiful sunset",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    width=1024,
    height=1024
)
```

---

**Version**: SDK Latest | **Python Only** | **Updated**: June 2026