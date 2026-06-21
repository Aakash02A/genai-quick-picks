# Mistral AI Python SDK

`pip install mistralai` · Python 3.9+

The official `mistralai` Python SDK provides a unified client to access Mistral's open and commercial models, offering chat completion, streaming, agents, embeddings, file management, and native OCR capabilities.

---

## Table of Contents

- [Client Initialization](#client-initialization)
- [Model Management](#model-management)
- [Chat Completion](#chat-completion)
- [Streaming](#streaming)
- [Async Usage](#async-usage)
- [Structured Outputs](#structured-outputs)
- [Tool & Function Calling](#tool--function-calling)
- [Agents API](#agents-api)
- [OCR API (Mistral OCR)](#ocr-api-mistral-ocr)
- [Embeddings](#embeddings)
- [Files API](#files-api)
- [Error Handling](#error-handling)

---

## Client Initialization

Initialize the client using standard environment variables or explicit API keys. Using the client as a context manager is recommended to manage the underlying connection pool.

### `class mistralai.Mistral`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `os.environ.get("MISTRAL_API_KEY")` | Your Mistral API key. |
| `endpoint` | `str` | `"https://api.mistral.ai"` | The API base URL endpoint. |
| `timeout` | `float` | `None` | Custom timeout in seconds for request transactions. |

```python
import os
from mistralai import Mistral

# 1. Standard Client Initialization (reads MISTRAL_API_KEY from environment)
client = Mistral()

# 2. Explicit Key & Custom Configuration
client_custom = Mistral(
    api_key="your-api-key-here",
    endpoint="https://api.mistral.ai",
    timeout=60.0
)

# 3. Recommended Resource Management via Context Manager
with Mistral() as client:
    # Operations are performed here and connection pool is cleaned up automatically
    pass
```

---

## Model Management

Query the list of available models.

### `client.models.list()`

```python
import os
from mistralai import Mistral

# List all available model IDs
with Mistral() as client:
    models = client.models.list()
    for model in models.data:
        print(f"Model ID: {model.id} (Created: {model.created})")
```

---

## Chat Completion

Submit message lists to get text generation completions.

### `client.chat.complete(...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | *Required* | Targeted model ID (e.g. `"mistral-large-latest"`, `"open-mixtral-8x22b"`). |
| `messages` | `list[dict]` | *Required* | Conversation history list. Message format: `{"role": "user"\|"assistant"\|"system", "content": "..."}`. |
| `temperature` | `float` | `0.7` | Randomness parameter (0.0 to 1.0). Lower is more deterministic. |
| `max_tokens` | `int` | `None` | Max limit on output tokens to generate. |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold. |
| `random_seed` | `int` | `None` | Set a seed value for deterministic outputs. |
| `safe_prompt` | `bool` | `False` | Toggles whether to inject a system prompt for content moderation. |
| `response_format` | `dict` | `None` | Set formatting requirements (e.g. `{"type": "json_object"}`). |

```python
from mistralai import Mistral

# Standard text generation
with Mistral() as client:
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": "You are a concise engineering assistant."},
            {"role": "user", "content": "Explain the difference between Mixtral and Mistral Large."}
        ],
        temperature=0.2,
        max_tokens=500
    )
    print(response.choices[0].message.content)
```

---

## Streaming

Iteratively receive content chunks in real-time.

### `client.chat.stream(...)`

```python
from mistralai import Mistral

with Mistral() as client:
    response_stream = client.chat.stream(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": "Write a short creative story about a clockmaker."}
        ]
    )
    
    for chunk in response_stream:
        # Extract text content delta from the stream chunk
        content_delta = chunk.data.choices[0].delta.content
        if content_delta is not None:
            print(content_delta, end="", flush=True)
    print()
```

---

## Async Usage

Perform concurrent, non-blocking completions and streams by utilizing the asynchronous client methods.

### `client.chat.complete_async(...)` & `client.chat.stream_async(...)`

```python
import asyncio
import os
from mistralai import Mistral

async def run_async_calls():
    # Asynchronous context manager manages HTTP resources cleanly
    async with Mistral() as client:
        
        # 1. Non-streaming async chat completion
        response = await client.chat.complete_async(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Give me a single-sentence startup idea."}]
        )
        print(f"Idea: {response.choices[0].message.content}\n")

        # 2. Streaming async chat completion
        stream_response = await client.chat.stream_async(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Describe entropy in two paragraphs."}]
        )
        
        async for chunk in stream_response:
            content_delta = chunk.data.choices[0].delta.content
            if content_delta is not None:
                print(content_delta, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(run_async_calls())
```

---

## Structured Outputs

Enforce JSON layouts or strict schema constraints using native JSON Mode or Pydantic parsing.

### 1. Direct Pydantic Schema Parsing (`client.chat.parse`)

The `.parse()` wrapper validates the output directly against a Pydantic `BaseModel`.

```python
import os
from pydantic import BaseModel, Field
from mistralai import Mistral

# Define the target structured output schema
class EmployeeExtract(BaseModel):
    name: str = Field(description="The employee's full name.")
    department: str = Field(description="The department they belong to.")
    skills: list[str] = Field(description="List of tools or programming skills.")

with Mistral() as client:
    response = client.chat.parse(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": "Alice Vance joined the Frontend Engineering team as a senior developer. She specializes in React, TypeScript, and CSS-in-JS."}
        ],
        response_format=EmployeeExtract
    )
    
    # Access parsed pydantic object natively
    data = response.choices[0].message.parsed
    print(f"Name: {data.name}")
    print(f"Department: {data.department}")
    print(f"Skills: {data.skills}")
```

### 2. Manual JSON Schema Mode

For low-level constraints, pass the raw JSON schema directly into `client.chat.complete`.

```python
from mistralai import Mistral

schema_spec = {
    "type": "json_schema",
    "json_schema": {
        "name": "skills_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "skills"]
        },
        "strict": True
    }
}

with Mistral() as client:
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": "Extract name and skills: Bob knows Rust and Go."}
        ],
        response_format=schema_spec
    )
    print(response.choices[0].message.content)
```

---

## Tool & Function Calling

Bind Python function layouts as tools for the model. The model decides when to run the function and parses input variables.

```python
from mistralai import Mistral

# 1. Define tool execution function signatures
def check_order_status(order_id: str) -> str:
    """Retrieve delivery status updates for a customer package.

    Args:
        order_id: The unique tracking identifier for the shipment.
    """
    if "123" in order_id:
        return "In Transit - Expected Delivery tomorrow"
    return "Delivered"

# 2. Formulate JSON Schema tool definitions
tools_declaration = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Retrieve delivery status updates for a customer package.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The unique tracking identifier"}
                },
                "required": ["order_id"]
            }
        }
    }
]

with Mistral() as client:
    user_prompt = "Can you look up where my package order #123-abc is?"
    
    # 3. Call model with tools
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools_declaration,
        tool_choice="auto"
    )
    
    # 4. Check if function call was proposed
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        call = tool_calls[0]
        if call.function.name == "check_order_status":
            import json
            args = json.loads(call.function.arguments)
            execution_result = check_order_status(**args)
            
            # 5. Return result to model to finalize response
            final_response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "user", "content": user_prompt},
                    response.choices[0].message, # Original assistant tool proposal
                    {
                        "role": "tool",
                        "name": call.function.name,
                        "content": execution_result,
                        "tool_call_id": call.id
                    }
                ]
            )
            print(final_response.choices[0].message.content)
```

---

## Agents API

Submit queries directly to your custom Agents created on the Mistral console.

### `client.agents.complete(...)` & `client.agents.stream(...)`

```python
from mistralai import Mistral

# Query a custom agent session
with Mistral() as client:
    response = client.agents.complete(
        agent_id="my-custom-agent-id-here",
        messages=[
            {"role": "user", "content": "Help me write a performance review template."}
        ]
    )
    print(response.choices[0].message.content)
```

---

## OCR API (Mistral OCR)

Extract high-fidelity markdown layout, text, headers, footers, and table contents from PDFs or images.

### `client.ocr.process(...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | *Required* | Must be set to `"mistral-ocr-latest"`. |
| `document` | `dict` | *Required* | Payload describing file source, e.g. `{"type": "document_url", "document_url": "..."}` or `{"type": "document_image", "image_url": "..."}`. |
| `include_image_base64` | `bool` | `False` | Toggles outputting structural image frames in base64 data. |
| `table_format` | `str` | `"markdown"` | Structure style for parsed tables: `"markdown"` or `"html"`. |

```python
import os
from mistralai import Mistral

with Mistral() as client:
    # 1. Execute OCR processing on a document URL
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": "https://arxiv.org/pdf/2310.06825.pdf" # Example PDF paper
        },
        table_format="markdown"
    )
    
    # 2. Extract parsed pages and markdown layouts
    for idx, page in enumerate(ocr_response.pages):
        print(f"--- Page {page.index} Markdown Output ---")
        print(page.markdown)
```

---

## Embeddings

Convert string inputs into numeric vector representations.

### `client.embeddings.create(...)`

```python
from mistralai import Mistral

with Mistral() as client:
    # Generate vector arrays
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[
            "Retrieval Augmented Generation workflows are very efficient.",
            "Text vectorization enables semantic search."
        ]
    )
    
    # Access dimensional array values
    vector_dims = response.data[0].embedding
    print(f"Embeddings Dimensions Length: {len(vector_dims)}")
```

---

## Files API

Manage files uploaded to the Mistral servers for fine-tuning or OCR tasks.

### `client.files.upload(...)` & `client.files.list()`

```python
from mistralai import Mistral

with Mistral() as client:
    # 1. Upload a document
    with open("training_examples.jsonl", "rb") as f:
        file_info = client.files.upload(
            file={"file_name": "training_examples.jsonl", "content": f.read()},
            purpose="fine-tune" # or "ocr"
        )
    print(f"File uploaded. ID: {file_info.id}")

    # 2. List all uploaded files
    all_files = client.files.list()
    for file in all_files.data:
        print(f"- {file.filename} (ID: {file.id}, Purpose: {file.purpose})")

    # 3. Delete file
    client.files.delete(file_id=file_info.id)
    print(f"Deleted file: {file_info.id}")
```

---

## Error Handling

Intercept exceptions gracefully using the SDK validation error schema structures.

### Exception Reference
- `models.SDKError`: Primary exception catch-all for standard 4xx and 5xx API errors.
- `models.HTTPValidationError`: Raised specifically on 422 HTTP responses indicating invalid parameters.

```python
from mistralai import Mistral, models

try:
    with Mistral() as client:
        # Invalid parameters or non-existent model name triggers exception
        response = client.chat.complete(
            model="mistral-nonexistent-model",
            messages=[{"role": "user", "content": "Hello!"}]
        )
except models.HTTPValidationError as e:
    print(f"Parameter structural validation error: {e}")
except models.SDKError as e:
    print(f"Mistral API network or authentication error: {e.status_code} - {e.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```
