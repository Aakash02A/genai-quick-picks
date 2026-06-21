# OpenAI Python SDK


# Installation

```bash
pip install openai
```

---

# Imports

```python
from openai import OpenAI
```

## Quick Navigation

- [Client Initialization](#client-initialization)
- [Chat Completions API](#chat-completions-api-legacy)
- [Responses API](#responses-api-modern)
- [Audio Modality](#audio-modality)
- [Image Generation](#image-generation)
- [Video Generation](#video-generation)
- [Embeddings](#embeddings)
- [Moderations](#moderations)
- [Conversations API](#conversations-api)
- [Assistants API](#assistants-api)
- [Realtime API](#realtime-api)
- [Agents SDK](#agents-sdk)
- [Files API](#files-api)
- [Uploads API](#uploads-api)
- [Batches API](#batches-api)
- [Fine-Tuning](#fine-tuning)
- [Models & Admin](#models)
- [Error Handling](#error-handling)

---

## Client Initialization

### `class OpenAI(...)`

<details>
<summary><strong>Synchronous client for the OpenAI API with HTTPX connection pooling</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `api_key` | `str` | `$OPENAI_API_KEY` | API key for authentication |
| `organization` | `str` | `$OPENAI_ORG_ID` | Organization ID for billing attribution |
| `project` | `str` | `$OPENAI_PROJECT_ID` | Project ID for resource isolation |
| `base_url` | `str` | `$OPENAI_BASE_URL` | Base URL for API endpoints |
| `timeout` | `float` | `10 min` | Request timeout threshold |
| `max_retries` | `int` | `2` | Automatic retry attempts |

```python
from openai import OpenAI

# 1. Initialize with default environment variables
client = OpenAI()

# 2. Explicit API key and organization
client = OpenAI(api_key="sk-...", organization="org-...")

# 3. Custom endpoint
client = OpenAI(base_url="http://localhost:8000/v1")

# 4. Custom timeout and retry settings
client_fast = client.with_options(timeout=5.0, max_retries=0)
```

</details>

---

### `class AsyncOpenAI(...)`

<details>
<summary><strong>Asynchronous client for high-concurrency operations</strong></summary>

All methods must be called with `await`. Optimal for concurrent batch processing.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `api_key` | `str` | `$OPENAI_API_KEY` | API key for authentication |
| `organization` | `str` | `$OPENAI_ORG_ID` | Organization ID for billing |
| `project` | `str` | `$OPENAI_PROJECT_ID` | Project ID for resource isolation |
| `timeout` | `float` | `10 min` | Request timeout threshold |
| `max_retries` | `int` | `2` | Automatic retry attempts |

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    
    message = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(message.choices[0].message.content)

asyncio.run(main())
```

</details>

---

### Workload Identity Providers

<details>
<summary><strong>Eliminate static API keys using cloud-native authentication</strong></summary>

| Provider | Function | Use Case |
|---|---|---|
| Kubernetes | `k8s_service_account_token_provider()` | K8s service accounts |
| Azure | `azure_managed_identity_token_provider()` | Azure VMs, Container Instances |
| GCP | `gcp_id_token_provider()` | Google Cloud resources |

```python
from openai import OpenAI, k8s_service_account_token_provider

# 1. Kubernetes service account
client = OpenAI(api_key=k8s_service_account_token_provider())

# 2. Azure Managed Identity
from openai import azure_managed_identity_token_provider
client = OpenAI(api_key=azure_managed_identity_token_provider())

# 3. Google Cloud Platform
from openai import gcp_id_token_provider
client = OpenAI(api_key=gcp_id_token_provider())
```

**Benefits:** Automatic token rotation, zero static credentials, compliance audit trail

</details>

---

## Chat Completions API (Legacy)

### `def client.chat.completions.create(...)`

<details>
<summary><strong>Generate text completions from conversation history (legacy interface)</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID: `"gpt-4o"`, `"gpt-4-turbo"`, `"gpt-3.5-turbo"` |
| `messages` | `list[dict]` | *Required* | Conversation: `[{"role": "user", "content": "..."}]` |
| `max_completion_tokens` | `int` | `None` | Hard cap on output tokens (reasoning + text) |
| `temperature` | `float` | `1.0` | Randomness: 0 (deterministic) to 2 (creative) |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `reasoning_effort` | `str` | `"medium"` | For o-series: `"low"`, `"medium"`, `"high"`, `"none"` |
| `tools` | `list[dict]` | `None` | Function definitions for tool calling |
| `response_format` | `dict` | `None` | `{"type": "json_object"}` or grammar |
| `stream` | `bool` | `False` | Enable streaming mode |

```python
from openai import OpenAI

client = OpenAI()

# 1. Basic completion
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is quantum computing?"}],
    max_completion_tokens=512
)
print(response.choices[0].message.content)

# 2. With reasoning (o-series)
response = client.chat.completions.create(
    model="o1",
    messages=[{"role": "user", "content": "Solve x^2 + 3x + 2 = 0"}],
    reasoning_effort="high",
    max_completion_tokens=2000
)

# 3. With tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=tools,
    tool_choice="auto"
)

# 4. JSON output constraint
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Return JSON with name and age"}],
    response_format={"type": "json_object"}
)

# 5. Streaming mode
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

</details>

---

## Responses API (Modern)

### `def client.responses.create(...)`

<details>
<summary><strong>Modern stateful interface with server-side session management</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID: `"gpt-5.5"`, `"gpt-4o"`, `"o1"` |
| `input` | `str \| list` | *Required* | Raw string or structured input items |
| `instructions` | `str` | `None` | System instructions (priority over user) |
| `store` | `bool` | `true` | Persist response server-side |
| `previous_response_id` | `str` | `None` | Reference parent response to chain turns |
| `reasoning` | `dict` | `None` | `{"effort": "low" \| "medium" \| "high"}` |
| `max_completion_tokens` | `int` | `None` | Total token budget (reasoning + output) |
| `max_output_tokens` | `int` | `None` | Limit visible text output only |
| `temperature` | `float` | `1.0` | Randomness scaling |

```python
from openai import OpenAI

client = OpenAI()

# 1. Basic response creation
response = client.responses.create(
    model="gpt-4o",
    input="Explain machine learning in one paragraph",
    max_output_tokens=256
)
print(response.output[0].text)

# 2. With reasoning
response = client.responses.create(
    model="o1",
    input="Prove that √2 is irrational",
    reasoning={"effort": "high"},
    max_completion_tokens=4000
)

# 3. Stateful conversation chaining
first = client.responses.create(
    model="gpt-4o",
    input="What is Python?",
    max_output_tokens=256,
    store=True
)
followup = client.responses.create(
    model="gpt-4o",
    input="Give me practical examples",
    previous_response_id=first.id,
    store=True
)

# 4. Structured JSON output
response = client.responses.create(
    model="gpt-4o",
    input="Extract name, age, city from text",
    text={"format": "json"},
    max_output_tokens=256
)

# 5. Streaming responses
stream = client.responses.create(
    model="gpt-4o",
    input="Tell me a story",
    stream=True
)
for event in stream:
    if event.type == "response.text.delta":
        print(event.delta, end="")
```

</details>

---

### `def client.responses.retrieve(response_id)` / `def client.responses.list()`

<details>
<summary><strong>Retrieve and list stored responses</strong></summary>

```python
from openai import OpenAI

client = OpenAI()

# 1. Retrieve specific response
response = client.responses.retrieve("resp_123abc")
print(f"ID: {response.id}, Status: {response.status}")

# 2. List all responses with pagination
responses = client.responses.list(limit=20)
for resp in responses.data:
    print(f"ID: {resp.id}, Created: {resp.created_at}")
```

</details>

---

## Audio Modality

### `def client.audio.speech.create(...)`

<details>
<summary><strong>Convert text to high-fidelity spoken audio</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | `"tts-1"` (fast), `"tts-1-hd"` (high quality) |
| `input` | `str` | *Required* | Text to synthesize (≤4,096 characters) |
| `voice` | `str` | *Required* | `"alloy"`, `"echo"`, `"fable"`, `"nova"`, `"shimmer"` |
| `response_format` | `str` | `"mp3"` | `"mp3"`, `"opus"`, `"aac"`, `"flac"`, `"wav"`, `"pcm"` |
| `speed` | `float` | `1.0` | Playback speed: 0.25 to 4.0 |

```python
from openai import OpenAI

client = OpenAI()

# 1. Basic speech synthesis
response = client.audio.speech.create(
    model="tts-1-hd",
    input="Hello, this is text-to-speech.",
    voice="nova"
)
response.write_to_file("output.mp3")

# 2. Custom format and speed
response = client.audio.speech.create(
    model="tts-1",
    input="Please speak quickly",
    voice="alloy",
    response_format="wav",
    speed=1.5
)

# 3. With word boundaries for lip-sync
response = client.audio.speech.create(
    model="tts-1-hd",
    input="Hello world",
    voice="nova",
    extra_body={"enable_word_boundary": True}
)
```

</details>

---

### `def client.audio.transcriptions.create(...)` / `def client.audio.translations.create(...)`

<details>
<summary><strong>Transcribe and translate audio using Whisper (99+ languages)</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file` | `BinaryIO` | *Required* | Audio file: mp3, wav, m4a, flac, opus, ogg |
| `model` | `str` | *Required* | `"whisper-1"` |
| `language` | `str` | `None` | ISO-639-1 code: `"en"`, `"fr"`, `"es"`, etc. |
| `response_format` | `str` | `"json"` | `"json"`, `"text"`, `"srt"`, `"vtt"`, `"verbose_json"` |
| `timestamp_granularities` | `list` | `None` | `["segment"]` or `["word"]` |

```python
from openai import OpenAI

client = OpenAI()

# 1. Basic transcription
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )
print(transcript.text)

# 2. With word-level timestamps
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

# 3. Specify language for accuracy
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="en"
    )

# 4. Translate foreign audio to English
with open("spanish_audio.mp3", "rb") as f:
    translation = client.audio.translations.create(
        model="whisper-1",
        file=f
    )
print(translation.text)  # English translation
```

</details>

---

## Image Generation

### `def client.images.generate(...)`

<details>
<summary><strong>Generate images from text prompts using DALL-E</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | `"dall-e-3"`, `"dall-e-2"` |
| `prompt` | `str` | *Required* | Detailed description (1,000+ chars recommended) |
| `size` | `str` | `"1024x1024"` | Dimensions: multiples of 16, aspect ≤3:1 |
| `quality` | `str` | `"standard"` | `"standard"`, `"hd"` (DALL-E 3 only) |
| `style` | `str` | `"vivid"` | `"vivid"`, `"natural"` (DALL-E 3 only) |
| `n` | `int` | `1` | Number of images to generate |
| `response_format` | `str` | `"url"` | `"url"` (24-hour expiry), `"b64_json"` |

```python
from openai import OpenAI
import base64

client = OpenAI()

# 1. Basic image generation
response = client.images.generate(
    model="dall-e-3",
    prompt="A serene landscape with mountains and lake at sunset",
    size="1024x1024"
)
print(response.data[0].url)

# 2. High-quality rendering
response = client.images.generate(
    model="dall-e-3",
    prompt="A detailed astronaut portrait",
    quality="hd",
    style="vivid"
)

# 3. Base64 output for embedding
response = client.images.generate(
    model="dall-e-3",
    prompt="A minimalist logo design",
    response_format="b64_json"
)
image_data = base64.b64decode(response.data[0].b64_json)

# 4. Multiple images
response = client.images.generate(
    model="dall-e-2",
    prompt="A robot in a garden",
    n=4,
    size="512x512"
)
```

</details>

---

### `def client.images.edit(...)` / `def client.images.create_variation(...)`

<details>
<summary><strong>Edit images via inpainting or create variations</strong></summary>

```python
from openai import OpenAI

client = OpenAI()

# 1. Inpaint masked regions
response = client.images.edit(
    image=open("base_image.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="Add a sunset in the background",
    model="dall-e-2"
)
print(response.data[0].url)

# 2. Create variations
response = client.images.create_variation(
    image=open("original.png", "rb"),
    n=4,
    size="1024x1024"
)
```

</details>

---

## Video Generation

### `def client.videos.create(...)` / `def client.videos.extend(...)` / `def client.videos.remix(...)`

<details>
<summary><strong>Generate and edit video using Sora</strong></summary>

⚠️ **Deprecation Notice:** Sora-2 series shutting down **September 24, 2026**

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | `"sora-2"`, `"sora-2-pro"` (DEPRECATED) |
| `prompt` | `str` | *Required* | Detailed video description |
| `input_reference` | `dict` | `None` | Image or video reference |
| `duration` | `int` | `None` | Duration in seconds (max 20) |

```python
from openai import OpenAI

client = OpenAI()

# 1. Text-to-video
response = client.videos.create(
    model="sora-2",
    prompt="A serene waterfall with golden light",
    duration=10
)
print(f"Video job: {response.id}")

# 2. Image-to-video
with open("input_image.jpg", "rb") as f:
    response = client.videos.create(
        model="sora-2",
        prompt="Zoom in smoothly",
        input_reference={"type": "image", "image": f},
        duration=10
    )

# 3. Extend video
extend_response = client.videos.extend(
    video_id=response.id,
    prompt="Continue the scene",
    duration=10
)

# 4. Remix (change theme/motion)
remix_response = client.videos.remix(
    video_id=response.id,
    prompt="Change to a winter scene with snow",
    duration=10
)

# 5. Check status
job = client.videos.retrieve(response.id)
print(f"Status: {job.status}")
```

</details>

---

## Embeddings

### `def client.embeddings.create(...)`

<details>
<summary><strong>Convert text to vector embeddings for semantic search and RAG</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | `"text-embedding-3-small"`, `"text-embedding-3-large"` |
| `input` | `str \| list[str]` | *Required* | String or list of strings to embed |
| `dimensions` | `int` | `None` | Truncate to dimensions (1–3,072) |
| `encoding_format` | `str` | `"float"` | `"float"`, `"base64"` |

```python
from openai import OpenAI

client = OpenAI()

# 1. Single string embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog"
)
embedding = response.data[0].embedding
print(f"Dimension: {len(embedding)}")

# 2. Batch embedding
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=[
        "Machine learning is powerful",
        "Deep learning uses neural networks"
    ]
)

# 3. Reduced dimensions for storage
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Important query",
    dimensions=512
)
```

</details>

---

## Moderations

### `def client.moderations.create(...)`

<details>
<summary><strong>Check content for policy violations (free service)</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | `"omni-moderation-latest"` |
| `input` | `str \| list[dict]` | *Required* | Text string or multimodal items |
| `modalities` | `list[str]` | `["text"]` | `["text"]`, `["image"]`, or both |

```python
from openai import OpenAI
import base64

client = OpenAI()

# 1. Text moderation
response = client.moderations.create(
    model="omni-moderation-latest",
    input="This is a test message"
)
if response.results[0].flagged:
    print("Content flagged")

# 2. Image moderation
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.moderations.create(
    model="omni-moderation-latest",
    input=[{"type": "image", "image": image_b64}]
)

# 3. Multimodal moderation
response = client.moderations.create(
    model="omni-moderation-latest",
    input=[
        {"type": "text", "text": "Check this:"},
        {"type": "image", "image": image_b64}
    ]
)
```

</details>

---

## Conversations API

### `def client.conversations.create(...)` / `def client.conversations.retrieve(...)` / `def client.conversations.list()`

<details>
<summary><strong>Server-side conversation storage for persistent multi-turn sessions</strong></summary>

```python
from openai import OpenAI

client = OpenAI()

# 1. Create conversation
conversation = client.conversations.create()
print(f"Conversation ID: {conversation.id}")

# 2. Use in Responses API
response = client.responses.create(
    model="gpt-4o",
    conversation_id=conversation.id,
    input="What is Python?"
)

# 3. Continue conversation (server maintains history)
followup = client.responses.create(
    model="gpt-4o",
    conversation_id=conversation.id,
    input="Give me practical examples"
)

# 4. Retrieve conversation
conv = client.conversations.retrieve(conversation.id)
print(f"Created: {conv.created_at}")

# 5. List conversations
conversations = client.conversations.list(limit=10)
```

</details>

---

## Assistants API

### `def client.beta.assistants.create(...)`

<details>
<summary><strong>Create AI assistant with tools, instructions, and resource access</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `name` | `str` | `None` | Display name |
| `description` | `str` | `None` | Description for organization |
| `instructions` | `str` | `None` | System instructions defining behavior |
| `model` | `str` | *Required* | Model: `"gpt-4o"`, `"gpt-4-turbo"` |
| `tools` | `list[dict]` | `None` | `"code_interpreter"`, `"file_search"`, functions |
| `tool_resources` | `dict` | `None` | Vector stores, sandbox configs |

```python
from openai import OpenAI

client = OpenAI()

# 1. Create with code interpreter
assistant = client.beta.assistants.create(
    name="Python Expert",
    instructions="You are an expert Python developer",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}]
)
print(f"Assistant ID: {assistant.id}")

# 2. Create with file search (RAG)
vector_store = client.beta.vector_stores.create()
assistant = client.beta.assistants.create(
    name="Document Assistant",
    model="gpt-4o",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {"vector_store_ids": [vector_store.id]}
    }
)

# 3. With custom tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]
assistant = client.beta.assistants.create(
    name="Weather Bot",
    model="gpt-4o",
    tools=tools
)
```

</details>

---

### `def client.beta.threads.create(...)` / `def client.beta.threads.messages.create(...)` / `def client.beta.threads.runs.create(...)`

<details>
<summary><strong>Create conversation threads and execute assistant runs</strong></summary>

```python
from openai import OpenAI
import time

client = OpenAI()

# 1. Create thread
thread = client.beta.threads.create()
print(f"Thread ID: {thread.id}")

# 2. Add message to thread
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What is 2 + 2?"
)

# 3. Create run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id="asst_..."
)

# 4. Poll for completion
while run.status not in ["completed", "failed"]:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread.id, run.id)

# 5. Get final messages
messages = client.beta.threads.messages.list(thread.id)
for msg in messages:
    if msg.role == "assistant":
        print(msg.content[0].text)

# 6. Stream run for real-time updates
with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id="asst_..."
) as stream:
    for text in stream.text_deltas:
        print(text, end="")
```

</details>

---

### `def client.beta.vector_stores.create(...)`

<details>
<summary><strong>Create vector store for automatic RAG and file search</strong></summary>

```python
from openai import OpenAI

client = OpenAI()

# 1. Create vector store
vector_store = client.beta.vector_stores.create(
    name="Company Docs"
)

# 2. Upload files to vector store
with open("document.pdf", "rb") as f:
    file_response = client.files.create(
        file=f,
        purpose="assistants"
    )

client.beta.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file_response.id
)

# 3. Link to assistant for RAG
assistant = client.beta.assistants.create(
    model="gpt-4o",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {"vector_store_ids": [vector_store.id]}
    }
)
```

</details>

---

## Realtime API

### `async def client.realtime.connect(...)`

<details>
<summary><strong>Ultra-low-latency bi-directional audio/text streaming</strong></summary>

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()
    
    # 1. Open realtime connection
    async with client.realtime.connect(
        model="gpt-4o-realtime-preview",
        modalities=["audio"],
        voice="nova"
    ) as conn:
        
        # 2. Configure session
        await conn.session.update(
            instructions="You are a helpful assistant"
        )
        
        # 3. Stream events
        async for event in conn:
            if event.type == "response.audio.delta":
                # Process audio chunk
                pass
            elif event.type == "response.text.delta":
                print(event.delta, end="")

asyncio.run(main())
```

</details>

---

## Agents SDK

### `class Agent(...)` / `class Runner(...)` / `class SandboxAgent(...)`

<details>
<summary><strong>Multi-agent orchestration with handoffs and sandboxing</strong></summary>

```python
from openai import Agent, Runner, SandboxAgent, Manifest

# 1. Create specialized agents
support_agent = Agent(
    instructions="You are a customer support specialist",
    tools=[{"type": "function", "function": {...}}]
)

billing_agent = Agent(
    instructions="You are a billing specialist"
)

# 2. Manager pattern (coordinator)
manager = Agent(
    instructions="Route to specialist agents",
    tools=[
        support_agent.as_tool(),
        billing_agent.as_tool()
    ]
)

# 3. Run agent with streaming
runner = Runner(agent=manager)
response = runner.run(input="I need billing help")
print(response)

# 4. Handoff pattern (decentralized routing)
triage_agent = Agent(
    instructions="Triage requests to specialists",
    handoffs=[support_agent, billing_agent]
)

# 5. Sandbox for code execution
manifest = Manifest(
    python="3.11",
    bash=True
)
code_agent = SandboxAgent(
    manifest=manifest,
    instructions="Execute Python scripts safely"
)
```

</details>

---

## Files API

### `def client.files.create(...)` / `def client.files.retrieve(...)` / `def client.files.delete(...)` / `def client.files.list()`

<details>
<summary><strong>Upload and manage files for Assistants, Batches, Fine-tuning</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file` | `BinaryIO` | *Required* | File object to upload |
| `purpose` | `str` | *Required* | `"batch"`, `"fine-tune"`, `"assistants"`, `"user_data"`, `"evals"` |
| `expires_after` | `dict` | `None` | Auto-delete policy: `{"anchor": "upload", "days": 30}` |

```python
from openai import OpenAI

client = OpenAI()

# 1. Upload for fine-tuning
with open("training.jsonl", "rb") as f:
    file = client.files.create(
        file=f,
        purpose="fine-tune"
    )
print(f"File ID: {file.id}")

# 2. Upload with expiration
with open("temp_data.jsonl", "rb") as f:
    file = client.files.create(
        file=f,
        purpose="batch",
        expires_after={"anchor": "upload", "days": 7}
    )

# 3. Retrieve file info
file = client.files.retrieve(file.id)
print(f"Status: {file.status}, Size: {file.bytes}")

# 4. Delete file
client.files.delete(file.id)

# 5. List files by purpose
files = client.files.list()
```

</details>

---

## Uploads API

### `def client.uploads.create(...)` / `def client.uploads.parts.create(...)` / `def client.uploads.complete(...)`

<details>
<summary><strong>Upload large files (>50MB) using chunked multipart transfer</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `filename` | `str` | *Required* | Target filename |
| `mime_type` | `str` | *Required* | File MIME type |
| `bytes` | `int` | `None` | Total file size |
| `purpose` | `str` | *Required* | `"fine-tune"`, `"batch"`, `"assistants"` |

```python
from openai import OpenAI

client = OpenAI()

# 1. Create upload session
upload = client.uploads.create(
    filename="large_dataset.jsonl",
    mime_type="application/jsonl",
    bytes=104857600,  # 100 MB
    purpose="fine-tune"
)
print(f"Upload ID: {upload.id}")

# 2. Upload file parts (automatic chunking)
with open("large_dataset.jsonl", "rb") as f:
    parts_response = client.uploads.parts.create(
        upload_id=upload.id,
        data=f
    )

# 3. Complete upload
final = client.uploads.complete(upload_id=upload.id)
file_id = final.file.id
print(f"File ready: {file_id}")
```

</details>

---

## Batches API

### `def client.batches.create(...)` / `def client.batches.retrieve(...)` / `def client.batches.cancel(...)` / `def client.batches.list()`

<details>
<summary><strong>Process up to 50,000 requests at 50% cost reduction</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `input_file_id` | `str` | *Required* | JSONL file ID with requests |
| `endpoint` | `str` | *Required* | API endpoint path |
| `completion_window` | `str` | `"24h"` | Processing deadline |
| `metadata` | `dict` | `None` | Custom tracking metadata |

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. Prepare batch requests (JSONL)
requests = [
    {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is AI?"}],
            "max_tokens": 100
        }
    }
]

with open("batch.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# 2. Upload batch file
batch_file = client.files.create(
    file=open("batch.jsonl", "rb"),
    purpose="batch"
)

# 3. Create batch job
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(f"Batch ID: {batch.id}")

# 4. Poll for completion
import time
while batch.status not in ["completed", "failed"]:
    time.sleep(30)
    batch = client.batches.retrieve(batch.id)

# 5. Download results
result = client.files.content(batch.output_file_id)
print(result.text)

# 6. Cancel batch (optional)
client.batches.cancel(batch.id)

# 7. List batches with pagination
batches = client.batches.list(limit=20)
```

</details>

---

## Fine-Tuning

### `def client.fine_tuning.jobs.create(...)`

<details>
<summary><strong>Fine-tune models with SFT, DPO, or RL for domain specialization</strong></summary>

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Base model: `"gpt-4o-mini"`, `"gpt-4-turbo"` |
| `training_file` | `str` | *Required* | File ID with training examples |
| `validation_file` | `str` | `None` | File ID for validation |
| `method` | `str` | `"sft"` | `"sft"` (Supervised), `"dpo"`, `"rl"` |
| `hyperparameters` | `dict` | `None` | Learning rate, batch size, epochs |

```python
from openai import OpenAI

client = OpenAI()

# 1. Upload training data (JSONL)
with open("training.jsonl", "rb") as f:
    training_file = client.files.create(
        file=f,
        purpose="fine-tune"
    )

# 2. Create fine-tuning job
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini",
    training_file=training_file.id,
    method="sft",
    hyperparameters={
        "learning_rate_multiplier": 1.0,
        "batch_size": 8,
        "n_epochs": 3
    }
)
print(f"Job ID: {job.id}")

# 3. Monitor with Weights & Biases
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini",
    training_file=training_file.id,
    integrations=[{
        "type": "wandb",
        "wandb": {"project": "my-project"}
    }]
)

# 4. Poll for completion
import time
while job.status not in ["succeeded", "failed"]:
    time.sleep(10)
    job = client.fine_tuning.jobs.retrieve(job.id)

if job.status == "succeeded":
    print(f"Model: {job.fine_tuned_model}")

# 5. List all jobs
jobs = client.fine_tuning.jobs.list(limit=10)

# 6. Stream training events
events = client.fine_tuning.jobs.list_events(job.id)

# 7. Cancel job
client.fine_tuning.jobs.cancel(job.id)
```

</details>

---

## Models

### `def client.models.list()` / `def client.models.retrieve(model_id)`

<details>
<summary><strong>List and inspect available models</strong></summary>

```python
from openai import OpenAI

client = OpenAI()

# 1. List all available models
models = client.models.list()
for model in models.data:
    print(f"{model.id} (owned by: {model.owned_by})")

# 2. Retrieve specific model details
model = client.models.retrieve("gpt-4o")
print(f"Created: {model.created}")
```

</details>

---

## Organization & Admin

### `def client.organization.admin_api_keys.list()` / `.retrieve()` / `.create()`

<details>
<summary><strong>Manage organization API keys (admin only)</strong></summary>

```python
from openai import OpenAI

client = OpenAI()

# 1. List API keys
api_keys = client.organization.admin_api_keys.list()
for key in api_keys.data:
    print(f"{key.name}: {key.created_at}")

# 2. Retrieve key details
key = client.organization.admin_api_keys.retrieve("key_...")

# 3. Create key with expiration
new_key = client.organization.admin_api_keys.create(
    name="Batch Processing",
    expires_at=1735689600  # Unix timestamp
)
```

</details>

---

## Error Handling

<details>
<summary><strong>Exception hierarchy and retry strategies</strong></summary>

| Exception | HTTP Code | Action |
|---|:---:|---|
| `AuthenticationError` | 401 | Check API key validity |
| `PermissionError` | 403 | Verify organization access |
| `NotFoundError` | 404 | Confirm resource exists |
| `RateLimitError` | 429 | Implement exponential backoff |
| `BadRequestError` | 400 | Review parameter validation |
| `InternalServerError` | 500+ | Retry with backoff |
| `APIConnectionError` | — | Check network connectivity |
| `APITimeoutError` | — | Increase timeout or use batches |

```python
from openai import OpenAI, RateLimitError, APIError
import time

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    print(f"Rate limited, retry after: {e.retry_after}")
    time.sleep(e.retry_after)
except APIError as e:
    print(f"API error: {e.status_code}")
```

</details>

---

## Platform Headers & Telemetry

<details>
<summary><strong>Response headers for monitoring and debugging</strong></summary>

| Header | Type | Purpose |
|---|---|---|
| `openai-organization` | `str` | Organization for billing attribution |
| `openai-processing-ms` | `int` | Server processing time (milliseconds) |
| `openai-version` | `str` | API version contract |
| `x-request-id` | `str` | Unique correlation ID for debugging |
| `x-ratelimit-remaining-requests` | `int` | Remaining request quota |
| `x-ratelimit-remaining-tokens` | `int` | Remaining token throughput quota |

</details>

---

## Migration: Chat Completions → Responses API

<details>
<summary><strong>Key differences and migration path</strong></summary>

| Aspect | Chat Completions | Responses API |
|---|---|---|
| **Session State** | Client-managed | Server-side persistence |
| **Max Tokens** | `max_tokens` (text only) | `max_completion_tokens` (total) |
| **Output** | Single message in choices | Typed output items |
| **Tool Execution** | Sequential callback | Decoupled tool/output items |
| **Reasoning Models** | Legacy `reasoning_effort` | Modern `reasoning` object |
| **Chaining** | Manual message appending | `previous_response_id` reference |

</details>

---

## Cost Optimization Strategies

<details>
<summary><strong>Reduce API costs and improve performance</strong></summary>

| Strategy | Benefit | Best For |
|---|---|---|
| **Batch Processing** | 50% discount | Non-real-time workloads |
| **Reasoning: Low** | Faster, fewer tokens | Simple tasks |
| **Token Counting** | Pre-execution budgeting | Cost forecasting |
| **Embedding Dimensions** | Storage efficiency | Vector databases |
| **Fine-tuning** | Domain optimization | Specialized use cases |
| **Prompt Caching** | Reduced token reuse | Repeated large contexts |

</details>

---

## Installation & Dependencies

<details>
<summary><strong>Package setup and optional extras</strong></summary>

```bash
# Core SDK
pip install openai

# With async support
pip install "openai[aiohttp]"

# Agents SDK
pip install "openai[agents]"

# Weights & Biases integration
pip install "openai[wandb]"
```

</details>

---

**Document Version:** 2024-Q4 | **SDK Version:** OpenAI Python >= 1.0.0
 
## Resources
 
- [API Reference](https://platform.openai.com/docs/api-reference)
- [openai-python on GitHub](https://github.com/openai/openai-python)
- [PyPI Package](https://pypi.org/project/openai/)
- [Agents SDK](https://github.com/openai/openai-agents-python)
 