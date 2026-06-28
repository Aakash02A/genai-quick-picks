# Anthropic (Claude) SDK Reference

**Latest Update**: June 2026  
**Official Docs**: [docs.anthropic.com](https://docs.anthropic.com)  
**GitHub**: [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)  
**Package**: `anthropic` v0.36+

---

## Table of Contents

- [Installation & Authentication](#installation--authentication)
- [Messages API — Core Generation](#messages-api--core-generation)
- [Streaming Responses](#streaming-responses)
- [Multimodal Input — Vision & Documents](#multimodal-input--vision--documents)
- [Tool Use (Function Calling)](#tool-use-function-calling)
- [Structured Outputs & JSON Mode](#structured-outputs--json-mode)
- [Prompt Caching — Cost Optimization](#prompt-caching--cost-optimization)
- [Batch API — Cost-Effective Processing](#batch-api--cost-effective-processing)
- [Token Counting & Cost Management](#token-counting--cost-management)
- [Error Handling & Retries](#error-handling--retries)
- [Computer Use (Beta)](#computer-use-beta)
- [Quick Reference](#quick-reference)
- [Production Best Practices](#production-best-practices)
- [Links & Resources](#links--resources)

---

## Installation & Authentication

### Package Installation

```bash
pip install anthropic --upgrade
# Requires Python 3.7+
```

### SDK Initialization

```python
import anthropic
import os

# Method 1: Auto-discover API key from environment
client = anthropic.Anthropic()
# Looks for ANTHROPIC_API_KEY environment variable

# Method 2: Explicit API key
client = anthropic.Anthropic(api_key="sk-ant-...")

# Method 3: Custom configuration
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    timeout=30.0,
    max_retries=3,
    base_url="https://api.anthropic.com"
)
```

### Async Client

```python
import anthropic

async_client = anthropic.AsyncAnthropic(api_key="sk-ant-...")
# Use with await for async operations
```

### Cloud Deployment Variants

| Deployment | Client Class | Configuration |
|:---|:---|:---|
| **Anthropic Platform** | `Anthropic()` | Standard SDK; set `ANTHROPIC_API_KEY` |
| **AWS Bedrock** | `AnthropicBedrock()` | AWS credentials; `aws_region="us-east-1"` |
| **Google Vertex AI** | `AnthropicVertex()` | GCP auth; `project_id="..."`, `region="us-central1"` |
| **Azure** | `AnthropicAzure()` | Azure credentials; `api_version="2024-06-01"` |

```python
# AWS Bedrock deployment
bedrock_client = anthropic.AnthropicBedrock(
    aws_region="us-east-1",
    aws_access_key="AKIAIOSFODNN7EXAMPLE",
    aws_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
)
```

---

## Messages API — Core Generation

### `client.messages.create(model, max_tokens, messages, ...)`

The primary API for generating text responses. Orchestrates tokenization, context building, and autoregressive decoding.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | Model ID: `"claude-opus-4-8"`, `"claude-sonnet-4-20250514"`, `"claude-haiku-4-8"` |
| `max_tokens` | `int` | *Required* | Maximum tokens in the response (required by API) |
| `messages` | `list[dict]` | *Required* | Message history; each has `role` and `content` |
| `system` | `str` | `None` | System prompt guiding model behavior |
| `temperature` | `float` | `1.0` | Randomness: `0.0` (deterministic) to `1.0` (creative) |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold; `0.1`–`1.0` |
| `top_k` | `int` | `None` | Top-K filtering; limits to top K tokens |
| `stop_sequences` | `list[str]` | `None` | Stop generation when these sequences appear |

**Unary (One-shot) Request**

```python
message = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Explain machine learning in simple terms"
        }
    ]
)

# Extract response text
response_text = message.content[0].text
print(f"Response: {response_text}")
print(f"Stop reason: {message.stop_reason}")  # "end_turn" or "stop_sequence"
print(f"Input tokens: {message.usage.input_tokens}")
print(f"Output tokens: {message.usage.output_tokens}")
```

**Multi-turn Conversation**

```python
messages = []

# Turn 1
messages.append({
    "role": "user",
    "content": "What is machine learning?"
})

response1 = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=messages
)

assistant_message = response1.content[0].text
messages.append({
    "role": "assistant",
    "content": assistant_message
})

print(f"Assistant: {assistant_message}\n")

# Turn 2 — context automatically preserved
messages.append({
    "role": "user",
    "content": "Can you explain neural networks?"
})

response2 = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=messages
)

print(f"Assistant: {response2.content[0].text}")
```

**System Prompt with Temperature Control**

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=512,
    system="You are a poetic writing assistant who responds in haikus.",
    temperature=0.7,  # Balanced creativity
    messages=[
        {"role": "user", "content": "Write about artificial intelligence"}
    ]
)

print(response.content[0].text)
```

---

## Streaming Responses

### `client.messages.stream(...)`

Streams tokens as they are generated for real-time feedback and reduced latency perception.

```python
# Method 1: Event-driven streaming
with client.messages.stream(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a short story about AI"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print("\n--- Stream complete ---")
```

**Accessing Stream Metadata**

```python
with client.messages.stream(
    model="claude-opus-4-8",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello, Claude"}]
) as stream:
    # Consume the stream
    for text in stream.text_stream:
        print(text, end="")
    
    # Access final message after stream completes
    final_message = stream.get_final_message()
    print(f"\nTotal tokens: {final_message.usage.output_tokens}")
    print(f"Stop reason: {final_message.stop_reason}")
```

**Async Streaming**

```python
import asyncio

async def stream_response():
    async with async_client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=512,
        messages=[{"role": "user", "content": "Tell me a joke"}]
    ) as stream:
        async for text in stream:
            if hasattr(text, 'delta') and text.delta.type == 'text_delta':
                print(text.delta.text, end="", flush=True)

asyncio.run(stream_response())
```

---

## Multimodal Input — Vision & Documents

### Image Processing

**From Base64-Encoded File**

```python
import base64

# Read and encode image
with open("photo.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail"
                }
            ]
        }
    ]
)

print(message.content[0].text)
```

**Supported Image Formats**

| Format | MIME Type | Max Size |
|:---|:---|:---|
| JPEG | `image/jpeg` | 20 MB |
| PNG | `image/png` | 20 MB |
| GIF | `image/gif` | 20 MB |
| WebP | `image/webp` | 20 MB |

### PDF Document Processing (Files API)

```python
# Upload PDF for analysis
with open("report.pdf", "rb") as f:
    response = client.beta.files.upload(file=f)
    file_id = response.id

# Use in message with beta feature flag
message = client.beta.messages.create(
    model="claude-opus-4-8",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": file_id
                    }
                },
                {
                    "type": "text",
                    "text": "Summarize the key findings from this PDF"
                }
            ]
        }
    ],
    betas=["files-api-2025-04-14"]
)

# Clean up
client.beta.files.delete(file_id)
print(message.content[0].text)
```

---

## Tool Use (Function Calling)

### Tool Definition & Execution Loop

```python
import json

# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Retrieves current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'San Francisco, CA'"
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
]

# Initial request with tools enabled
messages = [
    {"role": "user", "content": "What's the weather in Paris?"}
]

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

# Process tool calls in a loop
while response.stop_reason == "tool_use":
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None
    )
    
    if not tool_use_block:
        break
    
    tool_name = tool_use_block.name
    tool_input = tool_use_block.input
    
    print(f"Tool called: {tool_name}")
    print(f"Arguments: {json.dumps(tool_input, indent=2)}")
    
    # Execute the tool (in this example, simulated)
    if tool_name == "get_weather":
        tool_result = "Cloudy, 15°C, 60% humidity"
    else:
        tool_result = "Tool not found"
    
    # Continue conversation with tool result
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result
            }
        ]
    })
    
    # Get next response
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

# Extract final response
final_text = next(
    (block.text for block in response.content if block.type == "text"),
    None
)
print(f"Final response: {final_text}")
```

---

## Structured Outputs & JSON Mode

### JSON Response Format

```python
import json

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract person details: John Smith, 28 years old, Software Engineer"
        }
    ],
    extra_headers={
        "anthropic-beta": "structured-outputs-2025-04-14"
    }
)

# Parse JSON response
response_text = response.content[0].text
data = json.loads(response_text)
print(f"Name: {data.get('name')}")
print(f"Age: {data.get('age')}")
print(f"Job: {data.get('job')}")
```

---

## Prompt Caching — Cost Optimization

Reuse cached context across multiple requests to reduce costs and improve latency.

### Cache Configuration

```python
# First request establishes cache
response1 = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=512,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant."
        },
        {
            "type": "text",
            "text": "This is a long document that will be cached: " + "A" * 1000,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(f"Cache creation tokens: {response1.usage.cache_creation_input_tokens}")
print(f"Input tokens: {response1.usage.input_tokens}")

# Second request reuses cache
response2 = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=512,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant."
        },
        {
            "type": "text",
            "text": "This is a long document that will be cached: " + "A" * 1000,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ]
)

print(f"Cache read tokens: {response2.usage.cache_read_input_tokens}")  # Much lower!
print(f"Input tokens: {response2.usage.input_tokens}")
```

**Cache Metrics**

| Metric | Field | Calculation |
|:---|:---|:---|
| **Cache creation** | `cache_creation_input_tokens` | Tokens written to cache (first request) |
| **Cache hits** | `cache_read_input_tokens` | Tokens read from cache (subsequent requests) |
| **Cost savings** | `(cache_read * 0.1) / input` | 90% discount on cache reads vs. regular input |

---

## Batch API — Cost-Effective Processing

Process multiple messages efficiently at lower cost for non-urgent workloads.

### Batch Request Structure

```python
import anthropic
import time

client = anthropic.Anthropic()

# 1. Define batch requests
requests = [
    {
        "custom_id": "task-1",
        "params": {
            "model": "claude-opus-4-8",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "What is AI?"}
            ]
        }
    },
    {
        "custom_id": "task-2",
        "params": {
            "model": "claude-opus-4-8",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Explain machine learning"}
            ]
        }
    },
    {
        "custom_id": "task-3",
        "params": {
            "model": "claude-opus-4-8",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "What is deep learning?"}
            ]
        }
    }
]

# 2. Submit batch
batch = client.beta.messages.batches.create(
    requests=requests
)

print(f"Batch ID: {batch.id}")
print(f"Status: {batch.processing_status}")

# 3. Poll for completion
while batch.processing_status == "processing":
    time.sleep(5)
    batch = client.beta.messages.batches.retrieve(batch.id)
    print(f"Status: {batch.processing_status}")

# 4. Process results
results = client.beta.messages.batches.results(batch.id)
for result in results:
    print(f"Task {result.custom_id}: {result.result.message.content[0].text}")
```

**Batch Processing Characteristics**

| Aspect | Value |
|:---|:---|
| **Processing time** | 24 hours (typical) |
| **Cost reduction** | 50% discount on input tokens |
| **Max requests/batch** | No fixed limit |
| **Result expiration** | 29 days |

---

## Token Counting & Cost Management

### Token Estimation

```python
# Count input tokens before making a request
token_count = client.messages.count_tokens(
    model="claude-opus-4-8",
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)

print(f"Input tokens: {token_count.input_tokens}")

# With system prompt
token_count = client.messages.count_tokens(
    model="claude-opus-4-8",
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "Explain quantum computing in detail"}
    ]
)

print(f"Total tokens: {token_count.input_tokens}")
```

### Usage Tracking

```python
message = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

# Extract token usage
print(f"Input tokens: {message.usage.input_tokens}")
print(f"Output tokens: {message.usage.output_tokens}")

# Calculate cost (example pricing)
input_cost = message.usage.input_tokens * 0.003 / 1000
output_cost = message.usage.output_tokens * 0.015 / 1000
total_cost = input_cost + output_cost

print(f"Estimated cost: ${total_cost:.6f}")
```

**Model Pricing (as of June 2026)**

| Model | Input | Output |
|:---|:---|:---|
| `claude-opus-4-8` | $3/MTok | $15/MTok |
| `claude-sonnet-4-20250514` | $0.003/MTok | $0.015/MTok |
| `claude-haiku-4-8` | $0.0008/MTok | $0.004/MTok |

---

## Error Handling & Retries

### Exception Types

| Exception | HTTP Status | Handling |
|:---|:---|:---|
| `RateLimitError` | 429 | Implement exponential backoff |
| `APIConnectionError` | Network | Retry with exponential backoff |
| `APITimeoutError` | Timeout | Increase timeout; retry |
| `APIError` | Other | Log and escalate |

### Error Recovery Pattern

```python
import time
import anthropic

client = anthropic.Anthropic()

def call_with_retry(max_retries=3, base_delay=1.0):
    """Calls API with exponential backoff."""
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return message
        
        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
        
        except anthropic.APIConnectionError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Connection error. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise

# Use the function
message = call_with_retry()
print(message.content[0].text)
```

---

## Computer Use (Beta)

Anthropic provides an API to allow Claude to control a computer interface by viewing screenshots, moving the cursor, and clicking.

### Example Computer Use Request

```python
response = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1,
        }
    ],
    messages=[
        {"role": "user", "content": "Save a picture of a cat to my desktop."}
    ],
    betas=["computer-use-2024-10-22"]
)
print(response.content)
```

---

## Quick Reference

### Initialization

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")
# or: ANTHROPIC_API_KEY environment variable
```

### Basic Message

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.content[0].text)
```

### Streaming

```python
with client.messages.stream(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Story?"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

### Tool Use

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    tools=[{
        "name": "tool_name",
        "description": "...",
        "input_schema": {...}
    }],
    messages=[...]
)
```

### Token Counting

```python
tokens = client.messages.count_tokens(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": "..."}]
)
print(tokens.input_tokens)
```

### Batch Processing

```python
batch = client.beta.messages.batches.create(requests=[...])
results = client.beta.messages.batches.results(batch.id)
```

---

## Production Best Practices

1. **API Key Security**: Use environment variables; never hardcode keys
2. **Error Handling**: Catch `RateLimitError`, `APIConnectionError`, `APITimeoutError`
3. **Retry Strategy**: Implement exponential backoff for transient failures
4. **Token Management**: Monitor token usage; track costs
5. **Prompt Caching**: Enable for repeated context to reduce costs
6. **Streaming**: Use streaming for long responses to improve perceived latency
7. **Batch Processing**: Use batch API for non-urgent, high-volume requests
8. **Model Selection**: `opus-4-8` for reasoning, `sonnet` for balance, `haiku` for speed
9. **Timeout Handling**: Set appropriate timeouts; default is reasonable
10. **Monitoring**: Log all API calls and errors for debugging
11. **Tool Design**: Keep tool schemas simple and focused
12. **System Prompts**: Use system prompts to guide model behavior consistently

---

## Links & Resources

- **Official Docs**: https://docs.anthropic.com
- **API Reference**: https://docs.anthropic.com/reference
- **Python SDK**: https://github.com/anthropics/anthropic-sdk-python
- **Console**: https://console.anthropic.com
- **Cookbook**: https://github.com/anthropics/anthropic-cookbook
- **Models**: https://docs.anthropic.com/en/docs/about/models
- **Discord**: https://discord.gg/anthropic

---

**Version**: SDK 0.36+ | **Python Only** | **Updated**: June 2026