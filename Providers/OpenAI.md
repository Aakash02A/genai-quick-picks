# OpenAI SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [platform.openai.com/docs](https://platform.openai.com/docs)  
**GitHub**: [openai/openai-python](https://github.com/openai/openai-python)  
**Package**: `openai` (latest)

---

## Table of Contents

- [Installation & Authentication](#installation--authentication)
- [Chat Completions API](#chat-completions-api)
- [Streaming Responses](#streaming-responses)
- [Multimodal Input — Vision](#multimodal-input--vision)
- [File Management](#file-management)
- [Tool Use (Function Calling)](#tool-use-function-calling)
- [Structured Outputs — JSON Mode](#structured-outputs--json-mode)
- [Embeddings API](#embeddings-api)
- [Image Generation — DALL-E 3](#image-generation--dall-e-3)
- [Fine-tuning](#fine-tuning)
- [Batch Processing API](#batch-processing-api)
- [Error Handling](#error-handling)
- [Quick Reference](#quick-reference)
- [Production Best Practices](#production-best-practices)
- [Links & Resources](#links--resources)

---

## Installation & Authentication

### Package Installation

```bash
pip install openai --upgrade
# Requires Python 3.7+
```

### SDK Initialization

```python
from openai import OpenAI
import os

# Method 1: Auto-discover API key from environment
client = OpenAI()
# Looks for OPENAI_API_KEY environment variable

# Method 2: Explicit API key
client = OpenAI(api_key="sk-proj-...")

# Method 3: Custom configuration
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=30.0,
    max_retries=2,
    base_url="https://api.openai.com/v1"
)
```

### Async Client

```python
from openai import AsyncOpenAI

async_client = AsyncOpenAI(api_key="sk-proj-...")
# Use with await for async operations
```

### Azure OpenAI Deployment

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    api_version="2024-10-01-preview",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
)
```

---

## Chat Completions API

### `client.chat.completions.create(...)`

The primary API for text generation using the Messages format.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | Model ID: `"gpt-4o"`, `"gpt-4-turbo"`, `"gpt-3.5-turbo"` |
| `messages` | `list[dict]` | *Required* | Message history; each has `role` ("user", "assistant", "system") and `content` |
| `temperature` | `float` | `1.0` | Randomness: `0.0` (deterministic) to `2.0` (very creative) |
| `max_tokens` | `int` | `None` | Maximum response length; if unset, uses model default |
| `top_p` | `float` | `1.0` | Nucleus sampling; `0.0`–`1.0` |
| `frequency_penalty` | `float` | `0.0` | `-2.0` to `2.0`; penalizes repetition |
| `presence_penalty` | `float` | `0.0` | `-2.0` to `2.0`; encourages new topics |
| `seed` | `int` | `None` | Reproducibility seed for deterministic outputs |
| `stream` | `bool` | `False` | Enable token-by-token streaming |

**Simple Request**

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms"
        }
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
print(f"Stop reason: {response.choices[0].finish_reason}")
print(f"Total tokens used: {response.usage.total_tokens}")
```

**Multi-turn Conversation**

```python
messages = [
    {"role": "system", "content": "You are a Python coding expert."}
]

# Turn 1
messages.append({
    "role": "user",
    "content": "How do I read a JSON file in Python?"
})

response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

assistant_message = response1.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_message})
print(f"Assistant: {assistant_message}\n")

# Turn 2 — context preserved
messages.append({
    "role": "user",
    "content": "What about writing JSON to a file?"
})

response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

print(f"Assistant: {response2.choices[0].message.content}")
```

---

## Streaming Responses

### Stream Token-by-Token Output

```python
# Method 1: Iteration-based streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Write a short science fiction story"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Collecting Full Response from Stream**

```python
# Method 2: Accumulate stream into final message
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Write a haiku about AI"}
    ],
    stream=True
)

full_text = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        full_text += chunk.choices[0].delta.content

print(f"Complete response: {full_text}")
```

---

## Multimodal Input — Vision

### Image Input from File (Base64 Encoded)

```python
import base64

# Read and encode image
with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this architecture diagram"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        }
    ],
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Image from URL

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/1280px-Example.jpg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

**Supported Image Formats**

| Format | MIME Type | Max Resolution |
|:---|:---|:---|
| JPEG | `image/jpeg` | No limit |
| PNG | `image/png` | No limit |
| GIF | `image/gif` | No limit |
| WebP | `image/webp` | No limit |

---

## File Management

### Upload Files for Processing

```python
# Upload a file for use with API
with open("training_data.jsonl", "rb") as f:
    file_response = client.files.create(
        file=f,
        purpose="fine-tune"  # or "assistants", "batch"
    )

file_id = file_response.id
print(f"File uploaded: {file_id}")
print(f"File size: {file_response.size_bytes} bytes")

# List uploaded files
files = client.files.list()
for file in files.data:
    print(f"File: {file.id} ({file.filename})")

# Delete a file
client.files.delete(file_id)
print("File deleted")
```

---

## Tool Use (Function Calling)

### Tool Definition & Calling

```python
import json

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Initial request with tools
messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"}
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
        
        # Parse and execute
        args = json.loads(tool_call.function.arguments)
        
        if tool_call.function.name == "get_current_temperature":
            # Simulate function execution
            result = f"22°C, partly cloudy"
        else:
            result = "Function not found"
        
        # Continue conversation with result
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
        
        # Get final response
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )
        
        print(f"\nFinal response: {final_response.choices[0].message.content}")
```

---

## Structured Outputs — JSON Mode

### Enforce JSON Response Format

```python
import json

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Extract: John Doe, 28 years old, Software Engineer"
        }
    ],
    response_format={"type": "json_object"}
)

# Parse JSON response
data = json.loads(response.choices[0].message.content)
print(f"Name: {data.get('name')}")
print(f"Age: {data.get('age')}")
print(f"Job: {data.get('job')}")
```

---

## Embeddings API

### Generate Vector Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "This is a sample sentence",
        "Another sentence for comparison"
    ]
)

# Extract embeddings
for i, embedding in enumerate(response.data):
    print(f"Text {i}: {len(embedding.embedding)} dimensions")
    print(f"First 5 values: {embedding.embedding[:5]}")
```

**Embedding Models**

| Model | Dimension | Strengths |
|:---|:---|:---|
| `text-embedding-3-large` | 3072 | Highest quality, most expensive |
| `text-embedding-3-small` | 1536 | Balanced quality and cost |

### Semantic Search Example

```python
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

# Generate embeddings for documents
documents = [
    "The cat sat on the mat",
    "Dogs are loyal animals",
    "Python is a programming language"
]

doc_embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)

# Generate embedding for query
query = "What about cats?"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=[query]
)

# Find most similar document
query_vec = query_embedding.data[0].embedding
similarities = []

for i, doc_embed in enumerate(doc_embeddings.data):
    similarity = cosine_similarity(query_vec, doc_embed.embedding)
    similarities.append((documents[i], similarity))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
print(f"Most similar: {similarities[0][0]} (score: {similarities[0][1]:.3f})")
```

---

## Image Generation — DALL-E 3

### Generate Images from Text

```python
response = client.images.generate(
    model="dall-e-3",
    prompt="A serene mountain landscape at sunset with vibrant orange and purple colors",
    n=1,
    size="1024x1024",
    quality="hd",
    style="natural"
)

# Get image URL or B64
image_url = response.data[0].url
print(f"Image URL: {image_url}")

# Or save base64-encoded image
# import base64
# if response.data[0].b64_json:
#     with open("image.png", "wb") as f:
#         f.write(base64.b64decode(response.data[0].b64_json))
```

**DALL-E 3 Parameters**

| Parameter | Values | Default |
|:---|:---|:---|
| `size` | `1024x1024`, `1792x1024`, `1024x1792` | `1024x1024` |
| `quality` | `standard`, `hd` | `standard` |
| `style` | `natural`, `vivid` | `natural` |
| `n` | `1` | `1` (DALL-E 3 only supports 1 image per request) |

---

## Fine-tuning

### Create Fine-tuning Job

```python
import time

# Step 1: Upload training file (JSONL format)
with open("training_data.jsonl", "rb") as f:
    file_response = client.files.create(
        file=f,
        purpose="fine-tune"
    )

training_file_id = file_response.id

# Step 2: Create fine-tuning job
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini",
    training_file=training_file_id,
    hyperparameters={
        "learning_rate_multiplier": 1.0,
        "batch_size": 4,
        "n_epochs": 3
    }
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

# Step 3: Poll for completion
while job.status not in ["succeeded", "failed"]:
    time.sleep(10)
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job.status}")

if job.status == "succeeded":
    print(f"Fine-tuned model: {job.fine_tuned_model}")
    
    # Use the fine-tuned model
    response = client.chat.completions.create(
        model=job.fine_tuned_model,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
```

**Training Data Format (JSONL)**

```jsonl
{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
```

---

## Batch Processing API

### Process Multiple Requests Efficiently

```python
import time

# Step 1: Prepare batch requests (JSONL format)
batch_requests = [
    {
        "custom_id": "task-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is AI?"}],
            "max_tokens": 100
        }
    },
    {
        "custom_id": "task-2",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Explain ML"}],
            "max_tokens": 100
        }
    }
]

# Step 2: Write to file
import json
with open("batch_requests.jsonl", "w") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

# Step 3: Upload batch file
with open("batch_requests.jsonl", "rb") as f:
    batch_file = client.files.create(
        file=f,
        purpose="batch"
    )

# Step 4: Create batch
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

print(f"Batch ID: {batch.id}")

# Step 5: Poll for completion
while batch.status not in ["completed", "failed"]:
    time.sleep(30)
    batch = client.batches.retrieve(batch.id)
    print(f"Status: {batch.status}")

# Step 6: Retrieve results
if batch.status == "completed":
    results_file = batch.output_file_id
    results = client.files.content(results_file).text
    print(results)
```

**Batch Processing Costs**

| Aspect | Value |
|:---|:---|
| **Cost reduction** | 50% discount on input tokens |
| **Processing time** | ~24 hours (typically much faster) |
| **Expiration** | Results available for 24 hours |

---

## Error Handling

### Exception Types and Recovery

| Exception | HTTP Status | Strategy |
|:---|:---|:---|
| `RateLimitError` | 429 | Exponential backoff |
| `APIConnectionError` | Network | Retry with backoff |
| `APITimeoutError` | Timeout | Increase timeout; retry |
| `APIError` | Other | Log and escalate |

### Error Recovery Pattern

```python
import time
from openai import OpenAI, RateLimitError, APIConnectionError

client = OpenAI()

def call_with_retry(max_retries=3, base_delay=1.0):
    """Call API with exponential backoff on errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100
            )
            return response
        
        except RateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
        
        except APIConnectionError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Connection error. Waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise

# Use the function
response = call_with_retry()
print(response.choices[0].message.content)
```

---

## Quick Reference

### Initialization

```python
from openai import OpenAI

client = OpenAI(api_key="sk-proj-...")
# or: OPENAI_API_KEY environment variable
```

### Simple Chat

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Story?"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Vision

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
)
```

### Tool Use

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    tools=[...],
    tool_choice="auto"
)
```

### Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["text1", "text2"]
)
```

### Image Generation

```python
response = client.images.generate(
    model="dall-e-3",
    prompt="A beautiful sunset",
    size="1024x1024"
)
print(response.data[0].url)
```

---

## Production Best Practices

1. **API Key Security**: Use environment variables; never hardcode keys
2. **Error Handling**: Catch `RateLimitError`, `APIConnectionError`, `APITimeoutError`
3. **Retry Strategy**: Implement exponential backoff for transient failures
4. **Token Management**: Monitor token usage and costs
5. **Streaming**: Use streaming for long responses to improve UX
6. **Batch Processing**: Use batch API for non-urgent, high-volume requests (50% cost savings)
7. **Model Selection**: Latest models available for best performance
8. **Timeout Handling**: Set appropriate timeouts for your use case
9. **Monitoring**: Log all API calls, errors, and token usage
10. **Rate Limiting**: Implement request queuing and respect rate limits
11. **Tool Design**: Keep tool schemas simple and well-focused
12. **Vision Optimization**: Compress images where possible to reduce tokens

---

## Links & Resources

- **Official Docs**: https://platform.openai.com/docs
- **API Reference**: https://platform.openai.com/docs/api-reference
- **Python SDK**: https://github.com/openai/openai-python
- **Platform Console**: https://platform.openai.com
- **Cookbook**: https://github.com/openai/openai-cookbook
- **Models**: https://platform.openai.com/docs/models
- **Community**: https://community.openai.com

---

**Version**: SDK Latest | **Python Only** | **Updated**: June 2026