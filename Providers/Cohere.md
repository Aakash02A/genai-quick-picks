# Cohere SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [docs.cohere.com](https://docs.cohere.com)  
**GitHub**: [cohere-ai/cohere-python](https://github.com/cohere-ai/cohere-python)  
**Package**: `cohere` (V2)

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Chat API](#chat-api)
- [Embeddings](#embeddings)
- [Rerank API](#rerank-api)
- [Tool Use (Function Calling)](#tool-use-function-calling)
- [Fine-tuning](#fine-tuning)
- [Rerank V3](#rerank-v3)
- [Quick Reference](#quick-reference)

---

## Installation & Setup

### Package Installation

```bash
pip install cohere --upgrade
# Requires Python 3.7+
```

### Client Initialization

```python
import cohere
import os

# Method 1: Auto-discover API key
co = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY"))

# Method 2: Explicit configuration
co = cohere.ClientV2(
    api_key="your-api-key",
    timeout=30.0
)

# AWS Bedrock
co = cohere.BedrockClient(aws_region="us-east-1")

# Vertex AI (GCP)
co = cohere.VertexClient(project_id="your-project", region="us-central1")
```

---

## Chat API

### `co.chat(model, messages, ...)`

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | `"command-a-03-2025"`, `"command-r-plus"`, `"command-r"` |
| `messages` | `list[dict]` | *Required* | Message history with role and content |
| `temperature` | `float` | `1.0` | Randomness (0.0–2.0) |
| `max_tokens` | `int` | `None` | Maximum response length |
| `p` | `float` | `None` | Nucleus sampling |
| `k` | `int` | `None` | Top-K sampling |

```python
response = co.chat(
    model="command-a-03-2025",
    messages=[
        {"role": "user", "content": "Explain machine learning"}
    ]
)

print(response.message.content[0].text)
```

### Streaming

```python
stream = co.chat_stream(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "Write a story"}]
)

for event in stream:
    if hasattr(event, 'delta') and hasattr(event.delta, 'message'):
        print(event.delta.message.content[0].text, end="")
```

---

## Embeddings

### Generate Vector Embeddings

```python
response = co.embed(
    model="embed-english-v3.0",
    input_type="search_document",
    texts=["The quick brown fox", "Another sentence"]
)

for i, embedding in enumerate(response.embeddings):
    print(f"Text {i}: {len(embedding)} dimensions")
```

**Embedding Models**

| Model | Dimension | Language |
|:---|:---|:---|
| `embed-english-v3.0` | 1024 | English |
| `embed-multilingual-v3.0` | 1024 | 100+ languages |
| `embed-english-light-v3.0` | 384 | English (lightweight) |

---

## Rerank API

### Re-rank Documents by Relevance

```python
documents = [
    "The Eiffel Tower is in Paris, France",
    "The Statue of Liberty is in New York",
    "Big Ben is in London, England"
]

response = co.rerank(
    model="rerank-english-v3.0",
    query="Where is the Eiffel Tower?",
    documents=documents,
    top_n=2
)

for result in response.results:
    print(f"Doc {result.index}: {result.relevance_score:.2f}")
```

---

## Tool Use (Function Calling)

### Define and Execute Tools

```python
import json

tools = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations",
        "parameter_definitions": {
            "expression": {
                "description": "Mathematical expression",
                "type": "str",
                "required": True
            }
        }
    }
]

response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "What is 25 * 4 + 10?"}],
    tools=tools
)

if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Args: {tool_call.parameters}")
```

---

## Fine-tuning

### Create Fine-tuning Job

```python
import time

# Upload training file
with open("training_data.jsonl", "rb") as f:
    file_response = co.files.upload(file=f)

# Create job
job = co.fine_tuning.create(
    model="command-r-plus",
    training_file=file_response.id,
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 8,
        "num_epochs": 3
    }
)

print(f"Job ID: {job.id}")

# Poll for completion
while job.status == "PROCESSING":
    time.sleep(10)
    job = co.fine_tuning.get(job.id)

if job.status == "COMPLETED":
    print(f"Fine-tuned model: {job.fine_tuned_model_id}")
```

---

## Rerank V3

Cohere's Rerank V3 model provides improved ranking performance, especially for enterprise search and complex queries.

```python
response = co.rerank(
    model="rerank-english-v3.0",
    query="latest machine learning trends",
    documents=["Doc 1", "Doc 2"],
    top_n=2
)
```

---

## Quick Reference

```python
import cohere

co = cohere.ClientV2(api_key="your-key")

# Chat
response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.message.content[0].text)

# Stream
stream = co.chat_stream(model="command-a-03-2025", messages=[...])
for event in stream:
    if hasattr(event, 'delta'):
        print(event.delta.message.content[0].text, end="")

# Embeddings
embeddings = co.embed(
    model="embed-english-v3.0",
    input_type="search_document",
    texts=["text"]
)

# Rerank
reranked = co.rerank(
    model="rerank-english-v3.0",
    query="question",
    documents=["doc1", "doc2"]
)
```

---

**Version**: SDK V2 | **Python Only** | **Updated**: June 2026