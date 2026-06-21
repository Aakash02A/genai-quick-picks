# Anthropic Python SDK — Architecture & API Reference

> **Complete reference for the Core SDK, Agent SDK, and MCP SDK with collapsible sections.**

---

## Table of Contents

- [Core Claude SDK (`anthropic`)](#core-claude-sdk-anthropic)
- [Claude Agent SDK (`claude-agent-sdk`)](#claude-agent-sdk-claude-agent-sdk)
- [Model Context Protocol SDK (`mcp`)](#model-context-protocol-sdk-mcp)
- [Installation Extras](#installation-extras)
- [Exception Hierarchy Reference](#exception-hierarchy-reference)
- [Architecture Patterns](#architecture-patterns)
- [Feature Comparison Matrix](#feature-comparison-matrix)
- [Quick Start by Use Case](#quick-start-by-use-case)

---

## Core Claude SDK (`anthropic`)

> **Installation:** `pip install anthropic` (Python 3.9+)
>
> Primary Python SDK for accessing Anthropic's Claude models via the Messages API. Supports synchronous and asynchronous clients with platform-specific integrations for AWS, Google Cloud, and Azure.

---

<details>
<summary><strong>Client Classes</strong></summary>

---

<details>
<summary><code>class anthropic.Anthropic</code> — Synchronous client</summary>

Synchronous client for the Anthropic Messages API with automatic retry logic and connection pooling via httpx.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `api_key` | `str` | `None` | API key; defaults to `ANTHROPIC_API_KEY` env var |
| `timeout` | `float` | `600.0` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Maximum retries on transient errors (408, 409, 429, ≥500) |
| `transport` | `httpx.BaseTransport` | `None` | Custom HTTP transport; defaults to httpx connection pooling |

```python
from anthropic import Anthropic

# Default — reads ANTHROPIC_API_KEY from env
client = Anthropic()

# Explicit API key
client = Anthropic(api_key="sk-...")

# Custom timeout and retry settings
client = Anthropic(timeout=30.0, max_retries=5)

# Create a variant with different settings (non-destructive)
client_fast = client.with_options(timeout=5.0, max_retries=0)
```

</details>

---

<details>
<summary><code>class anthropic.AsyncAnthropic</code> — Asynchronous client</summary>

Asynchronous client for high-concurrency deployments — all methods must be called with `await`.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `api_key` | `str` | `None` | API key; defaults to `ANTHROPIC_API_KEY` env var |
| `timeout` | `float` | `600.0` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Maximum retries on transient errors |
| `transport` | `httpx.AsyncBaseTransport` | `None` | Custom async HTTP transport |

```python
import asyncio
from anthropic import AsyncAnthropic

async def main():
    client = AsyncAnthropic()

    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello Claude!"}]
    )
    print(message.content[0].text)

asyncio.run(main())
```

</details>

---

<details>
<summary><code>class anthropic.AnthropicBedrock</code> — AWS Bedrock</summary>

Routes requests through AWS Bedrock endpoints using SigV4 authentication or Bedrock Bearer Tokens.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `aws_region` | `str` | *Required* | AWS region code (e.g. `"us-east-1"`) |
| `aws_access_key` | `str` | `None` | AWS access key ID; defaults to `AWS_ACCESS_KEY_ID` env var |
| `aws_secret_key` | `str` | `None` | AWS secret access key; defaults to `AWS_SECRET_ACCESS_KEY` env var |
| `aws_session_token` | `str` | `None` | AWS session token for temporary credentials |
| `timeout` | `float` | `600.0` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Maximum retries |

```python
from anthropic import AnthropicBedrock

client = AnthropicBedrock(aws_region="us-west-2")

message = client.messages.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello from Bedrock!"}]
)
```

> **Install:** `pip install "anthropic[bedrock]"`

</details>

---

<details>
<summary><code>class anthropic.AnthropicVertex</code> — Google Cloud Vertex AI</summary>

Routes requests through GCP IAM-authenticated Vertex AI endpoints.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `project_id` | `str` | *Required* | Google Cloud project ID |
| `region` | `str` | `"us-central1"` | GCP region for the Vertex AI endpoint |
| `credentials` | `google.auth.credentials.Credentials` | `None` | GCP credentials; defaults to `GOOGLE_APPLICATION_CREDENTIALS` env var |
| `timeout` | `float` | `600.0` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Maximum retries |

```python
from anthropic import AnthropicVertex

client = AnthropicVertex(project_id="my-project", region="us-central1")

message = client.messages.create(
    model="claude-3-sonnet@20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello Vertex AI!"}]
)
```

> **Install:** `pip install "anthropic[vertex]"`

</details>

---

<details>
<summary><code>class anthropic.AnthropicFoundry</code> — Microsoft Foundry</summary>

Client for Microsoft Foundry — integrates managed credentials natively.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `organization` | `str` | `None` | Foundry organization identifier |
| `timeout` | `float` | `600.0` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Maximum retries |

</details>

---

<details>
<summary><code>class anthropic.AnthropicAWS</code> — AWS Marketplace</summary>

Client for Claude Platform on AWS Marketplace — uses IAM-based role authentication.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `role_arn` | `str` | *Required* | AWS IAM role ARN for authentication |
| `timeout` | `float` | `600.0` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Maximum retries |

> **Install:** `pip install "anthropic[aws]"`

</details>

</details>

---

<details>
<summary><strong>Messages API</strong></summary>

---

<details>
<summary><code>client.messages.create()</code> — Send a completion request</summary>

Sends a messages array to Claude and returns a completion response.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID (e.g. `"claude-3-5-sonnet-20241022"`) |
| `messages` | `list[MessageParam]` | *Required* | Conversation history as `{"role": "user"\|"assistant", "content": ...}` |
| `max_tokens` | `int` | *Required* | Hard cap on output tokens (range 1–200,000) |
| `system` | `str \| SystemPrompt` | `None` | System-level instructions |
| `temperature` | `float` | `1.0` | Randomness control (0–1; lower = focused, higher = creative) |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `top_k` | `int` | `None` | Limit sampling to top K tokens by probability |
| `stop_sequences` | `list[str]` | `None` | Stop generation when any sequence is produced |
| `stream` | `bool` | `False` | Enable raw delta-chunk streaming mode |
| `tools` | `list[Tool]` | `None` | Function definitions Claude can invoke |
| `tool_choice` | `ToolChoice` | `"auto"` | Tool-calling behavior: `"auto"`, `"any"`, `"none"`, or specific tool |
| `metadata` | `dict` | `None` | Arbitrary metadata to include with the request |
| `betas` | `list[str]` | `None` | Beta feature flags (e.g. `["files-api-2025-04-14"]`) |
| `timeout` | `float` | `None` | Override default timeout for this request only |
| `extra_headers` | `dict` | `None` | Additional HTTP headers |

```python
from anthropic import Anthropic

client = Anthropic()

# 1. Basic message
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is Python?"}]
)
print(message.content[0].text)

# 2. With system prompt and temperature
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    system="You are a creative storyteller.",
    temperature=0.8,
    messages=[{"role": "user", "content": "Tell me a short story."}]
)

# 3. With tool definitions
tools = [{
    "name": "get_weather",
    "description": "Fetch current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}]
)
```

</details>

---

<details>
<summary><code>client.messages.stream()</code> — Managed streaming context manager</summary>

Yields typed events and accumulates the complete response. Preferred over raw `stream=True`.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID |
| `messages` | `list[MessageParam]` | *Required* | Conversation history |
| `max_tokens` | `int` | *Required* | Hard cap on output tokens |
| `system` | `str` | `None` | System-level instructions |
| `temperature` | `float` | `1.0` | Randomness control |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `top_k` | `int` | `None` | Limit sampling to top K tokens |
| `stop_sequences` | `list[str]` | `None` | Stop generation sequences |
| `tools` | `list[Tool]` | `None` | Function definitions |
| `tool_choice` | `ToolChoice` | `"auto"` | Tool-calling behavior |
| `metadata` | `dict` | `None` | Request metadata |
| `betas` | `list[str]` | `None` | Beta feature flags |
| `timeout` | `float` | `None` | Request-specific timeout |
| `extra_headers` | `dict` | `None` | Additional HTTP headers |

```python
from anthropic import Anthropic

client = Anthropic()

# 1. Stream text output
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    messages=[{"role": "user", "content": "Tell me a story."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# 2. Get final message after streaming
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Count to 10."}]
) as stream:
    final_message = stream.get_final_message()
    print(final_message.usage)

# 3. Iterate over all typed events
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello"}]
) as stream:
    for event in stream:
        print(f"Event type: {event.type}")
```

</details>

---

<details>
<summary><code>client.messages.create(stream=True)</code> — Raw streaming</summary>

Raw streaming mode — yields delta chunks without automatic accumulation. Use `.stream()` instead when possible.

```python
from anthropic import Anthropic

client = Anthropic()

stream = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    stream=True,
    messages=[{"role": "user", "content": "Tell me a poem."}]
)

for event in stream:
    if event.type == "content_block_delta":
        if hasattr(event.delta, 'text'):
            print(event.delta.text, end="", flush=True)
    elif event.type == "message_stop":
        print("\n[Stream complete]")
```

</details>

---

<details>
<summary><code>client.messages.count_tokens()</code> — Estimate token usage</summary>

Estimates token usage without making an inference call — useful for cost planning.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `messages` | `list[MessageParam]` | *Required* | Conversation history to count |
| `model` | `str` | *Required* | Model ID |
| `system` | `str` | `None` | System prompt text to include in count |
| `tools` | `list[Tool]` | `None` | Tool definitions to include in count |

```python
from anthropic import Anthropic

client = Anthropic()

messages = [{"role": "user", "content": "Explain quantum computing in 100 words."}]

# Count tokens before sending
token_count = client.messages.count_tokens(
    model="claude-3-5-sonnet-20241022",
    messages=messages
)
print(f"Input tokens: {token_count.input_tokens}")

# Count with system prompt and tools
token_count = client.messages.count_tokens(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    system="You are an expert physicist.",
    tools=[{"name": "search", "description": "Search the web", "input_schema": {}}]
)
```

</details>

---

<details>
<summary><code>client.with_options()</code> — Create a modified client copy</summary>

Returns a modified copy of the client without altering the original instance.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `timeout` | `float` | `None` | Override default timeout for all requests on the new client |
| `max_retries` | `int` | `None` | Override automatic retry count |
| `headers` | `dict` | `None` | Additional HTTP headers |
| `api_key` | `str` | `None` | Override API key |
| `default_headers` | `dict` | `None` | Default headers for all requests |
| `default_query` | `dict` | `None` | Default query parameters for all requests |

```python
from anthropic import Anthropic

client = Anthropic()

# Low-latency variant
client_fast = client.with_options(timeout=5.0, max_retries=0)

# Client with custom headers
client_custom = client.with_options(headers={"X-Custom-Header": "my-value"})

# Use for a specific request
message = client_fast.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    messages=[{"role": "user", "content": "Quick response!"}]
)
```

</details>

</details>

---

<details>
<summary><strong>Utilities</strong></summary>

---

<details>
<summary><code>client.beta.files.upload()</code> — File Upload API</summary>

Uploads a document using the File Upload API. Requires `betas=["files-api-2025-04-14"]` on message calls.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file_path` | `str` | `None` | Path to a local file to upload |
| `file_contents` | `bytes` | `None` | Raw byte contents of a file |
| `file_obj` | `BinaryIO` | `None` | Open file object (e.g. `open("doc.pdf", "rb")`) |
| `mime_type` | `str` | `"application/octet-stream"` | MIME type of the uploaded file |

```python
from anthropic import Anthropic

client = Anthropic()

# 1. Upload a local file
file_response = client.beta.files.upload(file_path="document.pdf")
file_id = file_response.id

# 2. Use the uploaded file in a message
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize this document."},
            {"type": "document", "source": {"type": "file", "file_id": file_id}}
        ]
    }],
    betas=["files-api-2025-04-14"]
)

# 3. Upload from a binary stream
import io
pdf_bytes = io.BytesIO(b"PDF binary data...")
file_response = client.beta.files.upload(
    file_contents=pdf_bytes.read(),
    mime_type="application/pdf"
)
```

</details>

---

<details>
<summary><code>client.beta.messages.tool_runner()</code> — Iterative tool-calling loop</summary>

Executes tool-calling workflows iteratively — handles Claude's tool calls and returns results automatically.

```python
import asyncio
from anthropic import AsyncAnthropic

async def main():
    client = AsyncAnthropic()

    tools = [{
        "name": "calculator",
        "description": "Perform arithmetic operations",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }
    }]

    async def execute_tool(tool_name, tool_input):
        if tool_name == "calculator":
            a, b = tool_input["a"], tool_input["b"]
            op = tool_input["operation"]
            if op == "add":
                return a + b
            elif op == "multiply":
                return a * b
        return None

    result = await client.beta.messages.tool_runner(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": "What is 25 * 4?"}],
        tool_executor=execute_tool
    )
    print(result)

asyncio.run(main())
```

</details>

---

<details>
<summary><code>@beta_tool</code> — Decorator for auto-schema generation</summary>

Generates JSON schemas from Python function signatures and docstrings.

```python
from anthropic import Anthropic
from anthropic.lib._tools import beta_tool

@beta_tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a city.

    Args:
        city: Name of the city
        unit: Temperature unit ('celsius' or 'fahrenheit')
    """
    return f"Sunny, 22 {unit} in {city}"

# Extract the tool schema
tool_schema = get_weather.tool_definition()

# Use in a messages call
client = Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[tool_schema],
    messages=[{"role": "user", "content": "What's the weather?"}]
)
```

</details>

---

<details>
<summary><code>mcp_tool()</code> / <code>async_mcp_tool()</code> — MCP tool conversion</summary>

Converts an MCP-standard tool definition to the `BetaFunctionTool` format used by the Messages API.

```python
from anthropic import Anthropic
from anthropic.lib.tools import mcp_tool

mcp_definition = {
    "name": "search",
    "description": "Search the web",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }
}

tool = mcp_tool(mcp_definition)

client = Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[tool],
    messages=[{"role": "user", "content": "Search for Python tutorials"}]
)
```

**Async variant:**

```python
import asyncio
from anthropic import AsyncAnthropic
from anthropic.lib.tools import async_mcp_tool

async def main():
    tool = async_mcp_tool({"name": "search", "description": "Search"})
    client = AsyncAnthropic()
    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool],
        messages=[{"role": "user", "content": "Search for AI news"}]
    )

asyncio.run(main())
```

</details>

---

<details>
<summary><code>mcp_resource_to_content()</code> — MCP resource conversion</summary>

Converts an MCP resource definition to a message content block for inclusion in conversation context.

```python
from anthropic import Anthropic
from anthropic.lib.tools import mcp_resource_to_content

resource = {
    "uri": "schema://database_config",
    "name": "Database Configuration",
    "description": "Current database schema and credentials",
    "contents": [{"type": "text", "text": "Users table: id, email, created_at"}]
}

content_block = mcp_resource_to_content(resource)

client = Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Optimize this query:"},
            content_block
        ]
    }]
)
```

</details>

</details>

---

<details>
<summary><strong>Exception Hierarchy</strong></summary>

All exceptions derive from `APIError`. HTTP status codes map to specific exception classes.

| HTTP Code | Exception Class | Cause & Resolution |
|:---:|---|---|
| 400 | `BadRequestError` | Invalid request parameters or malformed payloads; review input schema |
| 401 | `AuthenticationError` | Missing, expired, or invalid API key; verify credentials |
| 403 | `PermissionDeniedError` | Insufficient permissions for the requested model or resource |
| 404 | `NotFoundError` | Invalid request URI or targeted model not found; check model ID |
| 409 | `ConflictError` | State conflict; automatically retried by default |
| 422 | `UnprocessableEntityError` | Well-formed request with semantic or schema validation errors |
| 429 | `RateLimitError` | Request volume limits exceeded; implement exponential backoff |
| ≥500 | `InternalServerError` | Server-side execution failure; automatically retried |
| — | `APIConnectionError` | Network connection failure or server unreachable |
| — | `APITimeoutError` | Client-side execution timeout exceeded (default 10 minutes) |
| — | `ValueError` | Raised if a non-streaming request exceeds 10 minutes without override |

```python
from anthropic import Anthropic
import anthropic

client = Anthropic()

try:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}]
    )
except anthropic.AuthenticationError as e:
    print(f"Auth failed: {e.message}")
except anthropic.RateLimitError:
    print("Rate limited — implement backoff")
except anthropic.APITimeoutError as e:
    print(f"Request timed out after {e.timeout} seconds")
except anthropic.APIConnectionError as e:
    print(f"Network error: {e.message}")
except anthropic.APIError as e:
    print(f"API error: {e.status_code} — {e.message}")
```

</details>

---

---

## Claude Agent SDK (`claude-agent-sdk`)

> **Installation:** `pip install claude-agent-sdk` (Python 3.10+)
>
> Stateful execution environment for building autonomous coding agents and interactive assistants. Bundles the Claude Code CLI binary and provides tools for filesystem operations, shell execution, and lifecycle hooks.

---

<details>
<summary><strong>Core API</strong></summary>

---

<details>
<summary><code>async def query()</code> — Single-task agent run</summary>

Asynchronous generator that executes a single-task agent run and yields iterative response messages.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `prompt` | `str` | *Required* | User instruction or task description |
| `cwd` | `str` | Current working directory | Root directory for the agent's workspace operations |
| `options` | `ClaudeAgentOptions` | `None` | Configuration object for tools, hooks, and permissions |

```python
import asyncio
from claude_agent_sdk import query

async def main():
    # Execute a simple task
    async for response in query("Create a Python script that prints 'Hello World'"):
        print(response)

    # Execute in a specific directory
    async for response in query(
        "Add type hints to the main.py file",
        cwd="/path/to/project"
    ):
        print(response)

asyncio.run(main())
```

</details>

---

<details>
<summary><code>class ClaudeSDKClient</code> — Stateful multi-turn client</summary>

Stateful, bidirectional interface for complex multi-turn agent sessions with context management and session forking.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `options` | `ClaudeAgentOptions` | `None` | Configuration object for tools, hooks, cwd, and permissions |

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        cwd="/path/to/project",
        allowed_tools=["Read", "Write", "Bash"]
    )
    client = ClaudeSDKClient(options=options)

    # Send a message and receive a response
    response = await client.send_message("Analyze the codebase structure")
    print(response.content)

    # Multi-turn conversation
    followup = await client.send_message("Now add unit tests for the main module")
    print(followup.content)

    # Fork the session for parallel exploration
    forked_client = client.fork()
    alt_response = await forked_client.send_message("Refactor for performance instead")

asyncio.run(main())
```

</details>

</details>

---

<details>
<summary><strong>Configuration &amp; Hooks</strong></summary>

---

<details>
<summary><code>class ClaudeAgentOptions</code> — Agent configuration</summary>

Controls agent behavior, tool access, and lifecycle hooks.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `cwd` | `str` | Current working directory | Root workspace directory |
| `allowed_tools` | `list[str]` | `None` | Tools that execute without prompting |
| `disallowed_tools` | `list[str]` | `None` | Tools that are never executed |
| `permission_mode` | `str` | `"acceptEdits"` | Default for unlisted tools: `"acceptEdits"`, `"ask"`, or `"deny"` |
| `hooks` | `dict[str, list[HookMatcher]]` | `None` | Lifecycle hooks for auditing and controlling tool execution |
| `mcp_servers` | `dict[str, MCP_Server]` | `None` | In-process MCP servers to extend agent capabilities |
| `cli_path` | `str` | `None` | Path to custom Claude Code CLI binary |

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, tool, create_sdk_mcp_server

# Basic security configuration
options = ClaudeAgentOptions(
    cwd="/home/user/projects",
    allowed_tools=["Read", "Write"],
    disallowed_tools=["Bash"],
    permission_mode="ask"
)

# Configuration with audit hooks
async def audit_bash(input_data, tool_use_id, context):
    command = input_data.get("tool_input", {}).get("command", "")
    if "rm -rf" in command:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Destructive commands blocked"
            }
        }
    return {}

options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],
    hooks={
        "PreToolUse": [HookMatcher(matcher="Bash", hooks=[audit_bash])]
    }
)
```

</details>

---

<details>
<summary><code>class HookMatcher</code> — Lifecycle hook callbacks</summary>

Registers lifecycle hook callbacks for specific tools or stages in the agent loop.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `matcher` | `str` | *Required* | Tool name (e.g. `"Bash"`) or stage identifier (e.g. `"SessionStart"`) |
| `hooks` | `list[Callable]` | *Required* | Callback functions executed when the matcher condition is met |

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

async def log_tool_use(input_data, tool_use_id, context):
    print(f"Tool: {input_data}, ID: {tool_use_id}")
    return {}

async def log_session_end(context):
    print(f"Session ended with status: {context.status}")
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="*", hooks=[log_tool_use])  # Match all tools
        ],
        "SessionEnd": [
            HookMatcher(matcher="*", hooks=[log_session_end])
        ]
    }
)
```

</details>

---

<details>
<summary><code>create_sdk_mcp_server()</code> — In-process MCP server</summary>

Creates an in-process MCP server wrapping custom Python functions as agent tools.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `name` | `str` | *Required* | Server name identifier |
| `version` | `str` | *Required* | Semantic version string (e.g. `"1.0.0"`) |
| `tools` | `list[Callable]` | *Required* | Functions decorated with `@tool` to expose |
| `resources` | `list[Callable]` | `None` | Functions decorated with `@resource` to expose as read-only data |

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("hash_file", "Compute SHA-256 for a file", {"filepath": str})
async def hash_file(args):
    import hashlib
    filepath = args.get("filepath")
    with open(filepath, "rb") as f:
        hash_obj = hashlib.sha256(f.read())
    return {"content": [{"type": "text", "text": hash_obj.hexdigest()}]}

@tool("validate_json", "Check if a file contains valid JSON", {"filepath": str})
async def validate_json(args):
    import json
    filepath = args.get("filepath")
    with open(filepath) as f:
        try:
            json.load(f)
            return {"content": [{"type": "text", "text": "Valid JSON"}]}
        except json.JSONDecodeError as e:
            return {"content": [{"type": "text", "text": f"Invalid JSON: {e}"}]}

server = create_sdk_mcp_server(
    name="file-tools",
    version="1.0.0",
    tools=[hash_file, validate_json]
)
```

</details>

---

<details>
<summary><code>@tool</code> — Register a Python function as an MCP tool</summary>

Decorator that registers a Python function as an MCP tool with automatic schema generation.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `name` | `str` | *Required* | Tool name identifier in the agent's context |
| `description` | `str` | *Required* | Human-readable description of what the tool does |
| `input_schema` | `dict` | *Required* | Dictionary mapping parameter names to their Python types |

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("greet", "Greet a person", {"name": str})
async def greet(args):
    name = args.get("name")
    return {"content": [{"type": "text", "text": f"Hello, {name}!"}]}

@tool("add_numbers", "Add two numbers", {"a": float, "b": float})
async def add_numbers(args):
    result = args.get("a") + args.get("b")
    return {"content": [{"type": "text", "text": str(result)}]}

@tool("parse_csv", "Parse a CSV file", {"filepath": str})
async def parse_csv(args):
    import csv
    filepath = args.get("filepath")
    with open(filepath) as f:
        rows = list(csv.DictReader(f))
    return {"content": [{"type": "text", "text": f"Parsed {len(rows)} rows"}]}

server = create_sdk_mcp_server(
    name="utilities",
    version="1.0.0",
    tools=[greet, add_numbers, parse_csv]
)
```

</details>

</details>

---

<details>
<summary><strong>Built-In System Tools</strong></summary>

| Tool | Operation | Primary Use Case |
|---|---|---|
| `Read` | Reads file contents | Code analysis and context gathering |
| `Write` | Creates new files | Code generation and file creation |
| `Edit` | Applies precise edits | Automated refactoring and bug patching |
| `Bash` | Runs shell commands | Testing, builds, and Git operations |
| `Monitor` | Monitors background scripts | Real-time logging and feedback |
| `Glob` | Locates files by pattern | Codebase navigation |
| `Grep` | Regex file search | Code pattern matching and audits |
| `WebSearch` | External web search | Gathering documentation and updates |
| `WebFetch` | Retrieves web pages | Reading API specifications |
| `AskUser` | Prompts user input | Clarifying requirements |

</details>

---

<details>
<summary><strong>Agent SDK Exceptions</strong></summary>

| Exception Class | Raised When | Recovery Strategy |
|---|---|---|
| `ClaudeSDKError` | Base exception for all Agent SDK errors | Catch parent class for blanket handling |
| `CLINotFoundError` | Claude Code CLI binary cannot be located | Verify `cli_path` or system installation |
| `CLIConnectionError` | SDK fails to establish connection with the runtime | Check network, subprocess limits, or port conflicts |
| `ProcessError` | Underlying process fails; exposes `exit_code` | Review `exit_code` for diagnostic information |
| `CLIJSONDecodeError` | Client fails to parse incoming JSON payloads | Verify CLI version compatibility |

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError, CLINotFoundError

async def safe_agent_run():
    try:
        client = ClaudeSDKClient()
        response = await client.send_message("Analyze code")
    except CLINotFoundError as e:
        print(f"CLI not found: {e}")
    except CLIConnectionError as e:
        print(f"Connection failed: {e}")
    except ProcessError as e:
        print(f"Process failed with exit code {e.exit_code}")
    except ClaudeSDKError as e:
        print(f"SDK error: {e}")
```

</details>

---

---

## Model Context Protocol SDK (`mcp`)

> **Installation:** `pip install mcp` (Python 3.10+)
>
> Python implementation of the Model Context Protocol — build interoperable MCP clients and servers that connect Claude with external tools, databases, and workflows.

---

<details>
<summary><strong>FastMCP Framework</strong></summary>

---

<details>
<summary><code>class mcp.server.fastmcp.FastMCP</code> — High-level server framework</summary>

High-level server framework using Python decorators to register tools, resources, and prompts.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `name` | `str` | *Required* | Server name identifier |
| `version` | `str` | `None` | Semantic version string (defaults to `"1.0.0"`) |

```python
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

mcp = FastMCP("ImageProcessor")

@mcp.tool()
def generate_thumbnail(image_path: str, size: int = 128) -> Image:
    """Create an optimized thumbnail for storage."""
    img = PILImage.open(image_path)
    img.thumbnail((size, size))
    return Image(data=img.tobytes(), format="png")

@mcp.resource("config://image_formats")
def list_image_formats() -> str:
    """Return supported image formats."""
    return "PNG, JPEG, WebP, GIF, SVG"

@mcp.prompt()
def analyze_image_quality(image_path: str) -> str:
    """Template for analyzing image quality."""
    return f"Analyze the quality of the image at {image_path}. Check resolution, compression artifacts, color balance."

if __name__ == "__main__":
    mcp.run()
```

</details>

</details>

---

<details>
<summary><strong>Decorators &amp; Primitives</strong></summary>

---

<details>
<summary><code>@mcp.tool()</code> — Register a Python function as an MCP tool</summary>

Registers a Python function as an MCP tool — automatically generates JSON schema from function signature.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `name` | `str` | `None` | Tool name; defaults to function name |
| `description` | `str` | `None` | Description; defaults to function docstring |

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DataTools")

# Simple tool with auto-generated schema
@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

# Explicit name and description
@mcp.tool(name="calculate", description="Perform arithmetic operations")
def math_op(a: float, b: float, operation: str) -> float:
    if operation == "add": return a + b
    elif operation == "subtract": return a - b
    elif operation == "multiply": return a * b
    elif operation == "divide": return a / b if b != 0 else 0

# Complex types
@mcp.tool()
def process_data(items: list[dict], filter_key: str) -> list[dict]:
    """Filter a list of dictionaries by key presence."""
    return [item for item in items if filter_key in item]
```

</details>

---

<details>
<summary><code>@mcp.resource()</code> — Register a read-only data source</summary>

Registers a Python function as an MCP resource — read-only data source loaded on demand.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `uri` | `str` | *Required* | Unique URI identifier (e.g. `"schema://database_layouts"`) |
| `name` | `str` | `None` | Resource name; defaults to function name |
| `description` | `str` | `None` | Description; defaults to function docstring |

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DatabaseServer")

@mcp.resource("schema://users")
def user_schema() -> str:
    """Returns the database schema for the users table."""
    return """
    Table: Users
    - id (INT PRIMARY KEY)
    - email (VARCHAR UNIQUE)
    - created_at (TIMESTAMP)
    """

@mcp.resource("config://search_settings")
def search_config() -> str:
    """Configuration for search operations."""
    import json
    return json.dumps({"max_results": 100, "timeout_seconds": 30, "index_type": "fulltext"}, indent=2)

@mcp.resource("status://server_health")
def server_health() -> str:
    """Current server health status."""
    import psutil
    return f"CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%"
```

</details>

---

<details>
<summary><code>@mcp.prompt()</code> — Register a reusable instruction template</summary>

Registers a Python function as an MCP prompt — reusable instruction template for the model.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `name` | `str` | `None` | Prompt name; defaults to function name |
| `description` | `str` | `None` | Description; defaults to function docstring |

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AnalysisTools")

@mcp.prompt()
def code_review() -> str:
    """Template for conducting code reviews."""
    return """Review the provided code for:
    1. Code style and consistency
    2. Performance inefficiencies
    3. Security vulnerabilities
    4. Test coverage gaps
    5. Documentation completeness"""

@mcp.prompt()
def optimize_query(slow_query: str) -> str:
    """Suggest indexing and optimization strategies for a slow SQL query."""
    return f"Analyze this database query and suggest optimization strategies:\n\n{slow_query}\n\nProvide specific indexing recommendations and query rewrites."

@mcp.prompt()
def security_audit(file_path: str) -> str:
    """Audit a file for security vulnerabilities."""
    return f"Perform a security audit on {file_path}. Check for: hardcoded credentials, unsafe deserialization, SQL injection, insecure randomness, buffer overflow risks."
```

</details>

</details>

---

<details>
<summary><strong>Low-Level Protocol</strong></summary>

---

<details>
<summary><code>class mcp.server.Server</code> — Low-level protocol server</summary>

Low-level protocol server for implementing custom transport mechanisms or advanced use cases.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `name` | `str` | *Required* | Server name identifier |
| `version` | `str` | `None` | Semantic version string |
| `tools` | `dict` | `None` | Dictionary of tool definitions |
| `resources` | `dict` | `None` | Dictionary of resource definitions |
| `prompts` | `dict` | `None` | Dictionary of prompt definitions |
| `transport` | `Transport` | `None` | Custom transport layer (stdio, SSE, WebSocket) |

```python
from mcp.server import Server, Tool, Resource
from mcp.types import TextContent

server = Server(name="CustomServer", version="1.0.0")

# Register a tool manually
tool = Tool(name="get_time", description="Get the current Unix timestamp",
            inputSchema={"type": "object", "properties": {}})
server.tools[tool.name] = tool

# Register a resource manually
resource = Resource(uri="time://now", name="CurrentTime", description="The current time")
server.resources[resource.uri] = resource

@server.call_tool
async def handle_tool_call(name: str, arguments: dict):
    if name == "get_time":
        import time
        return [TextContent(type="text", text=str(int(time.time())))]

@server.read_resource
async def handle_resource_read(uri: str):
    if uri == "time://now":
        import datetime
        return [TextContent(type="text", text=str(datetime.datetime.now()))]
```

</details>

</details>

---

<details>
<summary><strong>Integration with Claude</strong></summary>

```python
from anthropic import Anthropic
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP("WebTools")

@mcp_server.tool()
def fetch_url(url: str) -> str:
    """Fetch and return the contents of a web page."""
    import requests
    response = requests.get(url, timeout=10)
    return response.text[:5000]

client = Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[{
        "name": "fetch_url",
        "description": "Fetch and return the contents of a web page.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    }],
    messages=[{
        "role": "user",
        "content": "What's the title of the page at https://example.com?"
    }]
)
```

</details>

---

---

## Installation Extras

<details>
<summary><strong>Available Extras</strong></summary>

| Extra | Installation Command | Use Case |
|---|---|---|
| Base | `pip install anthropic` | Core Messages API client on Claude Platform |
| Bedrock | `pip install "anthropic[bedrock]"` | AWS Bedrock integration with SigV4 auth |
| Vertex | `pip install "anthropic[vertex]"` | Google Cloud Vertex AI integration |
| AWS | `pip install "anthropic[aws]"` | AWS Marketplace Claude Platform |
| Async HTTP | `pip install "anthropic[aiohttp]"` | High-concurrency async loops (replaces httpx) |
| MCP | `pip install "anthropic[mcp]"` | Native MCP translation utilities |

</details>

---

---

## Architecture Patterns

<details>
<summary><strong>Standard Orchestration (Sequential Tool Calls)</strong></summary>

- Model generates tool calls
- Client executes tools sequentially via network
- Results fed back into conversation

**Trade-offs:** ✅ Granular control &nbsp;|&nbsp; ❌ High context overhead &nbsp;|&nbsp; ❌ Higher latency (sequential round-trips)

```python
from anthropic import Anthropic

client = Anthropic()
messages = []

# Initial request with tools
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[],  # tool definitions here
    messages=[{"role": "user", "content": "Analyze the data"}]
)
messages.append({"role": "assistant", "content": response.content})

# Process tool calls and feed back results
for block in response.content:
    if block.type == "tool_use":
        tool_result = execute_tool(block.name, block.input)
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": block.id, "content": tool_result}]
        })

# Continue conversation
response2 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[],  # tool definitions here
    messages=messages
)
```

</details>

---

<details>
<summary><strong>Programmatic Tool Calling (PTC) — Local Code Execution</strong></summary>

- Model generates a single Python script
- Script executes locally with parallel tool runs
- Only final output sent back to model

**Trade-offs:** ✅ Up to 92% token reduction &nbsp;|&nbsp; ✅ Single network round-trip &nbsp;|&nbsp; ✅ Preserves privacy (no intermediate data leakage)

```python
import asyncio
from anthropic import AsyncAnthropic

async def main():
    client = AsyncAnthropic()

    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": """Generate a Python script that:
            1. Fetches user data from database
            2. Fetches order data from database
            3. Merges and filters results
            4. Returns consolidated output

            Use asyncio.gather() for parallel operations."""
        }]
    )

    script = response.content[0].text
    exec(script)

asyncio.run(main())
```

</details>

---

---

## Feature Comparison Matrix

<details>
<summary><strong>SDK Feature Comparison</strong></summary>

| Feature | Core SDK | Agent SDK | MCP SDK | Legacy Bedrock |
|---|:---:|:---:|:---:|:---:|
| **Python Version** | 3.9+ | 3.10+ | 3.10+ | 3.7+ |
| **Lifecycle** | ✅ Active | ✅ Active | ✅ Active | ❌ Deprecated |
| **Stateless Messages API** | ✅ | — | — | ✅ |
| **Stateful Agent Loop** | — | ✅ | — | — |
| **Tool Protocol** | ✅ JSON Schema | ✅ MCP-based | ✅ MCP standard | ✅ Completions |
| **Async Support** | ✅ AsyncAnthropic | ✅ All methods | ✅ Full | ✅ Limited |
| **Streaming** | ✅ SSE | ✅ Integrated | ✅ Stdio/SSE | ✅ SSE |
| **Bedrock Support** | ✅ Platform subclass | ✅ Yes | — | ✅ Native |
| **Vertex AI Support** | ✅ Platform subclass | ✅ Yes | — | — |
| **Custom Tools** | ✅ JSON schema | ✅ @tool decorator | ✅ @mcp.tool() | — |
| **Built-In System Tools** | — | ✅ 10+ tools | — | — |
| **Context Optimization** | Via token counting | Via local execution | Via local servers | — |

</details>

---

---

## Quick Start by Use Case

<details>
<summary><strong>Choose the right SDK for your use case</strong></summary>

| Use Case | Recommended | Why |
|---|---|---|
| **High-throughput API** | Core SDK | Stateless, minimal overhead |
| **Coding assistant** | Agent SDK | Built-in file/shell tools, stateful |
| **External tool integration** | MCP SDK | Standard protocol, interoperable |
| **AWS Bedrock** | Core SDK + `[bedrock]` | Modern interface, consolidated |
| **Legacy systems** | `anthropic-bedrock` | Only if Python < 3.9 required |

</details>

---

## Resources

- [Claude SDK Documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)
- [GitHub — claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python)
- [Claude Agent SDK Docs](https://code.claude.com/docs/en/agent-sdk/python)