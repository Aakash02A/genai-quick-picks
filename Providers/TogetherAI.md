# Together AI SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [docs.together.ai](https://docs.together.ai)  
**GitHub**: [togethercomputer/together-python](https://github.com/togethercomputer/together-python), [togethercomputer/together-typescript](https://github.com/togethercomputer/together-typescript)

## Table of Contents

- [Installation](#installation)
- [SDK Initialization](#sdk-initialization)
- [Authentication](#authentication)
- [Chat Completions (Core Generation)](#chat-completions-core-generation)
- [Available Models](#available-models)
- [Image Generation](#image-generation)
- [Embeddings](#embeddings)
- [Reranking](#reranking)
- [Fine-tuning](#fine-tuning)
- [File Management](#file-management)
- [Error Handling](#error-handling)
- [Advanced: Safety Features](#advanced-safety-features)
- [Production Best Practices](#production-best-practices)
- [Quick Reference](#quick-reference)
- [Links & Resources](#links--resources)

---

## Installation

### Python
```bash
pip install together --upgrade
# Requires Python 3.10+
```

### TypeScript/JavaScript
```bash
npm install together-ai
# Requires Node.js 20+ or TypeScript 4.9+
```

---

## SDK Initialization

### Python
```python
import os
from together import Together

# Basic initialization
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Or use environment variable directly
# export TOGETHER_API_KEY="your-key"
client = Together()

# With custom configuration
client = Together(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    timeout=30.0
)
```

### Python - Async
```python
from together import AsyncTogether

async_client = AsyncTogether(api_key="your-key")
# Use with await for async operations
```

### TypeScript/JavaScript
```typescript
import Together from "together-ai";

// Basic initialization
const client = new Together({
  apiKey: process.env.TOGETHER_API_KEY
});

// Or use environment variable
// export TOGETHER_API_KEY="your-key"
const client = new Together();

// With custom configuration
const client = new Together({
  apiKey: process.env.TOGETHER_API_KEY,
  timeout: 30000,
  baseURL: "https://api.together.xyz/v1"  // Optional custom URL
});
```

---

## Authentication

Together AI uses **API key authentication** via the `Authorization: Bearer` header.

```bash
# Set environment variable
export TOGETHER_API_KEY="your-api-key"
```

Get your API key from [Together AI Console](https://www.together.ai/console).

---

## Chat Completions (Core Generation)

### Python - Unary (Simple)
```python
from together import Together

client = Together()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[
        {
            "role": "user",
            "content": "Explain machine learning in simple terms"
        }
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Python - Streaming
```python
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[
        {"role": "user", "content": "Write a poem about AI"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Python - Async
```python
import asyncio
from together import AsyncTogether

async def main():
    client = AsyncTogether()
    
    response = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Python - Async Batch (Multiple Concurrent Requests)
```python
import asyncio
from together import AsyncTogether

async def main():
    client = AsyncTogether()
    
    prompts = [
        "What is AI?",
        "Explain quantum computing",
        "Define machine learning"
    ]
    
    tasks = [
        client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        for prompt in prompts
    ]
    
    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        print(f"Response {i}: {response.choices[0].message.content}")

asyncio.run(main())
```

### TypeScript - Unary
```typescript
import Together from "together-ai";

const client = new Together({
  apiKey: process.env.TOGETHER_API_KEY
});

const response = await client.chat.completions.create({
  model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  messages: [
    {
      role: "user",
      content: "Explain machine learning in simple terms"
    }
  ],
  temperature: 0.7,
  maxTokens: 1024
});

console.log(response.choices[0].message.content);
```

### TypeScript - Streaming
```typescript
const stream = await client.chat.completions.create({
  model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
  messages: [
    { role: "user", content: "Write a poem about AI" }
  ],
  stream: true
});

for await (const chunk of stream) {
  if (chunk.choices[0].delta.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
}

// Cancel stream if needed
stream.controller.abort();
```

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier (see below) |
| `messages` | array | Message history with `role` and `content` |
| `temperature` | float | 0.0-2.0; higher = more creative (default: 0.7) |
| `max_tokens` | int | Maximum response length (default: 512) |
| `top_p` | float | Nucleus sampling (0-1) |
| `top_k` | int | Top-K sampling |
| `repetition_penalty` | float | Penalize repetition |
| `stop` | list | Stop generation at these tokens |
| `safety_model` | string | Safety filter (e.g., "Meta-Llama/Llama-Guard-7b") |
| `stream` | bool | Enable streaming (default: false) |

---

## Available Models

Together AI hosts 200+ open-source and proprietary models. Key models:

### Chat/Text Models
| Model | Context | Type | Speed |
|-------|---------|------|-------|
| `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | 128K | Base | Very Fast |
| `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | 128K | Small | Ultra Fast |
| `mistralai/Mistral-7B-Instruct-v0.2` | 32K | Small | Very Fast |
| `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO` | 32K | Mixture | Fast |
| `teknium/OpenHermes-2p5-Mistral-7B` | 4K | Small | Very Fast |

### Code Models
| Model | Context | Language |
|-------|---------|----------|
| `meta-llama/CodeLlama-70b-Instruct-hf` | 100K | Multi-language |
| `meta-llama/CodeLlama-13b-Instruct-hf` | 16K | Multi-language |

### Specialized Models
| Model | Purpose |
|-------|---------|
| `mistralai/Mixtral-8x22B` | Mixture of experts, higher quality |
| `WizardLM/WizardLM-70B-V1` | General instruction-following |
| `lmsys/vicuna-13b-v1.5` | Chat optimized |

---

## Image Generation

### Python
```python
from together import Together

client = Together()

# Generate image
image = client.images.generate(
    prompt="A serene mountain landscape at sunset",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    width=1024,
    height=1024,
    steps=20,
    seed=42
)

# Save image
import base64

for i, img in enumerate(image.data):
    with open(f"image_{i}.png", "wb") as f:
        f.write(base64.b64decode(img.b64_json))

# Or use URL
for img in image.data:
    print(f"Image URL: {img.url}")
```

### TypeScript
```typescript
const image = await client.images.generate({
  prompt: "A serene mountain landscape at sunset",
  model: "stabilityai/stable-diffusion-xl-base-1.0",
  width: 1024,
  height: 1024,
  steps: 20,
  seed: 42
});

// Access B64 JSON
image.data.forEach((img, i) => {
  console.log(`Image ${i} (B64): ${img.b64Json.substring(0, 50)}...`);
});

// Or URL (if available)
image.data.forEach((img) => {
  console.log(`Image URL: ${img.url}`);
});
```

### Image Models

| Model | Type | Quality |
|-------|------|---------|
| `stabilityai/stable-diffusion-xl-base-1.0` | SDXL | High quality, versatile |
| `black-forest-labs/FLUX.1-dev` | Flux | State-of-the-art |
| `black-forest-labs/FLUX.1-schnell` | Flux | Fast variant |
| `stabilityai/stable-diffusion-3.5-large` | SD 3.5 | Advanced quality |

---

## Embeddings

Generate vector embeddings for semantic search and RAG.

### Python
```python
from together import Together

client = Together()

response = client.embeddings.create(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    input=[
        "This is a sample sentence.",
        "This is another sentence."
    ]
)

for i, embedding in enumerate(response.data):
    print(f"Text {i}: {len(embedding.embedding)} dimensions")
    print(f"First 5 values: {embedding.embedding[:5]}")
```

### TypeScript
```typescript
const response = await client.embeddings.create({
  model: "togethercomputer/m2-bert-80M-8k-retrieval",
  input: [
    "This is a sample sentence.",
    "This is another sentence."
  ]
});

response.data.forEach((embedding, i) => {
  console.log(`Text ${i}: ${embedding.embedding.length} dimensions`);
  console.log(`First 5 values: ${embedding.embedding.slice(0, 5)}`);
});
```

### Embedding Models

| Model | Dimension | Language |
|-------|-----------|----------|
| `togethercomputer/m2-bert-80M-8k-retrieval` | 768 | English |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | English |

---

## Reranking

Re-rank documents based on relevance to a query.

### Python
```python
from together import Together

client = Together()

documents = [
    "The Eiffel Tower is in Paris, France",
    "The Statue of Liberty is in New York",
    "Big Ben is in London, England"
]

response = client.rerank.create(
    model="Salesforce/Llama-Rank-V1",
    query="Where is the Eiffel Tower?",
    documents=documents,
    top_n=2
)

for result in response.results:
    print(f"Document {result.index}: {result.relevance_score:.4f}")
```

### TypeScript
```typescript
const response = await client.rerank.create({
  model: "Salesforce/Llama-Rank-V1",
  query: "Where is the Eiffel Tower?",
  documents: [
    "The Eiffel Tower is in Paris, France",
    "The Statue of Liberty is in New York",
    "Big Ben is in London, England"
  ],
  topN: 2
});

response.results.forEach((result) => {
  console.log(`Document ${result.index}: ${result.relevanceScore.toFixed(4)}`);
});
```

---

## Fine-tuning

Train custom models on your data.

### Python
```python
from together import Together

client = Together()

# Step 1: Upload training data (JSONL format)
# File format: {"text": "training example"}
with open("training_data.jsonl", "rb") as f:
    uploaded_file = client.files.upload(file=f)

print(f"File ID: {uploaded_file.id}")

# Step 2: Create fine-tuning job
job = client.fine_tuning.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    training_file=uploaded_file.id,
    n_epochs=3,
    learning_rate=0.001,
    batch_size=8,
    # wandb_api_key="optional-for-logging"  # Log to Weights & Biases
)

print(f"Fine-tuning job: {job.id}")

# Step 3: Monitor job status
import time

while True:
    job_status = client.fine_tuning.get(job.id)
    print(f"Status: {job_status.status}")
    
    if job_status.status in ["COMPLETED", "FAILED"]:
        break
    
    time.sleep(10)

# Step 4: Use fine-tuned model
if job_status.status == "COMPLETED":
    response = client.chat.completions.create(
        model=job_status.id,  # Use job ID as model
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
```

### TypeScript
```typescript
// Upload training data
const uploadedFile = await client.files.upload({
  file: fs.createReadStream("training_data.jsonl")
});

console.log(`File ID: ${uploadedFile.id}`);

// Create fine-tuning job
const job = await client.fineTuning.create({
  model: "meta-llama/Meta-Llama-3-8B-Instruct",
  trainingFile: uploadedFile.id,
  nEpochs: 3,
  learningRate: 0.001,
  batchSize: 8
});

console.log(`Fine-tuning job: ${job.id}`);

// Monitor status
let jobStatus = job;
while (jobStatus.status !== "COMPLETED" && jobStatus.status !== "FAILED") {
  await new Promise(resolve => setTimeout(resolve, 10000));
  jobStatus = await client.fineTuning.get(job.id);
  console.log(`Status: ${jobStatus.status}`);
}

// Use fine-tuned model
if (jobStatus.status === "COMPLETED") {
  const response = await client.chat.completions.create({
    model: jobStatus.id,
    messages: [{ role: "user", content: "Hello" }]
  });
  console.log(response.choices[0].message.content);
}
```

### Training Data Format (JSONL)
```jsonl
{"text": "Example 1: This is training data"}
{"text": "Example 2: Another training example"}
{"text": "Example 3: More training data"}
```

---

## File Management

### Python
```python
from together import Together

client = Together()

# Upload file
file = client.files.upload(
    file=open("document.txt", "rb"),
    purpose="fine-tuning"  # or "assistants", etc.
)

print(f"File ID: {file.id}")

# List files
files = client.files.list()
for f in files.data:
    print(f"File: {f.id} ({f.bytes} bytes)")

# Get file info
file_info = client.files.retrieve(file.id)
print(f"Created: {file_info.created_at}")

# Delete file
client.files.delete(file.id)
print("File deleted")
```

### TypeScript
```typescript
// Upload file
const file = await client.files.upload({
  file: fs.createReadStream("document.txt"),
  purpose: "fine-tuning"
});

console.log(`File ID: ${file.id}`);

// List files
const files = await client.files.list();
files.data.forEach((f) => {
  console.log(`File: ${f.id} (${f.bytes} bytes)`);
});

// Get file info
const fileInfo = await client.files.retrieve(file.id);
console.log(`Created: ${fileInfo.createdAt}`);

// Delete file
await client.files.delete(file.id);
console.log("File deleted");
```

---

## Error Handling

### Python
```python
from together import Together
from together.exceptions import APIConnectionError, RateLimitError

try:
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Implement exponential backoff
except APIConnectionError as e:
    print(f"Connection error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### TypeScript
```typescript
try {
  const response = await client.chat.completions.create({
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages: [{ role: "user", content: "Hello" }]
  });
} catch (error) {
  if (error.status === 429) {
    console.error("Rate limited");
    // Implement exponential backoff
  } else if (error.status >= 500) {
    console.error("Server error");
  } else {
    console.error(`Error: ${error.message}`);
  }
}
```

---

## Advanced: Safety Features

### Python
```python
# Use safety filter
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[
        {"role": "user", "content": "Generate a creative story"}
    ],
    safety_model="Meta-Llama/Llama-Guard-7b"
)

# Check safety flags in response
if hasattr(response, 'safety_flags'):
    print(f"Safety flags: {response.safety_flags}")
```

---

## Production Best Practices

1. **Model Selection**: Start with Llama 3.1 models for best balance
2. **Streaming**: Use streaming for real-time user feedback
3. **Async**: Use async client for concurrent requests
4. **Rate Limiting**: Implement backoff for 429 responses
5. **Timeouts**: Set appropriate request timeouts
6. **Error Handling**: Catch and handle errors gracefully
7. **Fine-tuning**: Train on domain-specific data for better results
8. **Monitoring**: Log all API calls for debugging
9. **Cost**: Use smaller models when possible to reduce costs
10. **Security**: Keep API keys secure; use environment variables
11. **Batch Processing**: Use async/concurrent requests for high volume
12. **Safety**: Use safety filters for production applications

---

## Quick Reference

### Most Common Operations

```python
# Python: Basic chat
from together import Together
client = Together()
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)

# Python: Streaming
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[...],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# Python: Embeddings
embeddings = client.embeddings.create(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    input=["text1", "text2"]
)

# Python: Image generation
image = client.images.generate(
    prompt="A beautiful sunset",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    width=1024,
    height=1024
)
```

```typescript
// TypeScript: Basic chat
import Together from "together-ai";
const client = new Together();
const response = await client.chat.completions.create({
  model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  messages: [{ role: "user", content: "Hello" }]
});
console.log(response.choices[0].message.content);

// TypeScript: Streaming
const stream = await client.chat.completions.create({
  model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  messages: [...],
  stream: true
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0].delta.content || "");
}

// TypeScript: Embeddings
const embeddings = await client.embeddings.create({
  model: "togethercomputer/m2-bert-80M-8k-retrieval",
  input: ["text1", "text2"]
});

// TypeScript: Image generation
const image = await client.images.generate({
  prompt: "A beautiful sunset",
  model: "stabilityai/stable-diffusion-xl-base-1.0",
  width: 1024,
  height: 1024
});
```

### Response Structure
```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "response text"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Environment Setup
```bash
export TOGETHER_API_KEY="your-api-key"
```

---

## Links & Resources

- **Official Docs**: https://docs.together.ai
- **Python SDK**: https://github.com/togethercomputer/together-python
- **TypeScript SDK**: https://github.com/togethercomputer/together-typescript
- **Console**: https://www.together.ai/console
- **Models**: https://www.together.ai/models
- **Pricing**: https://www.together.ai/pricing
- **Community**: https://together.ai/community
- **Blog**: https://www.together.ai/blog

---

**Version**: SDK 1.0+ | **Updated**: June 2026