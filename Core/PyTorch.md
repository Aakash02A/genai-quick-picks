# Pytorch GenAI SDK

# Installation

## 1. CPU-Only Installation
*Use this if you do not have a dedicated NVIDIA graphics card and only intend to run models on your computer's main processor.*

### Using Pip:
```bash
pip3 install torch torchvision torchaudio
```
### Using Conda
```bash
install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 2. GPU Installation (NVIDIA CUDA)

*If you have a compatible NVIDIA graphics card, ensure you have the correct [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) installed first. Match your command to your local CUDA Toolkit version.*

### Using Pip (For CUDA 12.1 / 12.4):

```bash
pip3 install torch torchvision torchaudio --index-url https://pytorch.org
```
### Using Conda (For CUDA 12.1 / 12.4)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Apple Silicon Mac Installation (M1 / M2 / M3 / M4)
*PyTorch supports accelerated GPU training natively on Apple Silicon chips through the Metal Performance Shaders (MPS) backend.*

### Using Pip:
```bash
pip3 install torch torchvision torchaudio
```

## Table of Contents

- [PyTorch Compute-Level SDK](#pytorch-compute-level-sdk)
  - [Native Attention & Dispatch Kernels](#native-attention--dispatch-kernels)
  - [Distributed Parallelism & Model Sharding](#distributed-parallelism--model-sharding)
  - [Graph Compilation & Backend Fusion](#graph-compilation--backend-fusion)
  - [Telemetry, Logging & Post-Training Frameworks](#telemetry-logging--post-training-frameworks)
- [Client-Service Integration SDK](#client-service-integration-sdk)
  - [Client Lifecycle & Authentication](#client-lifecycle--authentication)
  - [Messages Interface & Streaming](#messages-interface--streaming)
  - [Batch Operations & Async Queue Processing](#batch-operations--async-queue-processing)
  - [Extended Thinking & Token Budgets](#extended-thinking--token-budgets)
  - [Functional Tool Execution & Agent Control](#functional-tool-execution--agent-control)

---

---

# PyTorch Compute-Level SDK

> Low-level execution layer managing tensor operations, distributed sharding, attention kernels, and compiler-driven graph fusion for transformer inference and training.

---

## Native Attention & Dispatch Kernels

<details>
<summary><strong>torch.nn.functional.scaled_dot_product_attention — SDPA dispatch</strong></summary>

---

### `def torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, *, enable_flash, enable_mem_efficient, enable_math, enable_cudnn)`

Computes scaled dot-product attention over query, key, and value tensors, dynamically dispatching to the most efficient hardware backend.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `query` | `Tensor` | *Required* | Query projection of shape `(B, H, L, E)` representing active token contexts |
| `key` | `Tensor` | *Required* | Key projection of shape `(B, H, S, E)` representing sequence history |
| `value` | `Tensor` | *Required* | Value projection of shape `(B, H, S, E_v)` containing contextual vectors |
| `attn_mask` | `Tensor / None` | `None` | Optional binary or additive mask; `True` positions **are permitted** to attend |
| `dropout_p` | `float` | `0.0` | Dropout probability applied to the normalized attention matrix |
| `is_causal` | `bool` | `False` | Enforces causal masking for autoregressive loops, skipping unneeded computations |
| `scale` | `float / None` | `None` | Explicit scale factor; defaults to `1 / sqrt(d_k)` when `None` |
| `enable_flash` | `bool` | `True` | Toggles the FlashAttention backend for GPU memory bandwidth optimization |
| `enable_mem_efficient` | `bool` | `True` | Activates Memory-Efficient Attention for intermediate sequence lengths |
| `enable_math` | `bool` | `True` | Activates the baseline PyTorch C++ implementation for debugging |
| `enable_cudnn` | `bool` | `True` | Activates cuDNN-specific pathways on compatible NVIDIA enterprise platforms |

> ⚠️ **Mask polarity difference:** In `torch.nn.Transformer`, `True` **blocks** a position. In `scaled_dot_product_attention`, `True` **permits** attention. Invert masks when migrating legacy code through SDPA.

> ℹ️ **AMD hardware:** When `enable_flash=True`, PyTorch defaults to AOTriton. To override to Composable Kernel (CK), set `TORCH_ROCM_FA_PREFER_CK=1` in the environment.

```python
import torch
import torch.nn.functional as F

# 1. Define tensor dimensions
B, H, L, S, E = 2, 8, 128, 128, 64

# 2. Create query, key, value projections
query = torch.randn(B, H, L, E, device="cuda", dtype=torch.float16)
key   = torch.randn(B, H, S, E, device="cuda", dtype=torch.float16)
value = torch.randn(B, H, S, E, device="cuda", dtype=torch.float16)

# 3. Run SDPA with causal masking (autoregressive decoding)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)

# 4. Restrict to a specific backend (e.g., FlashAttention only)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_mem_efficient=False,
    enable_math=False,
    enable_cudnn=False,
):
    output_flash = F.scaled_dot_product_attention(query, key, value, is_causal=True)

print(output.shape)  # torch.Size([2, 8, 128, 64])
```

---

<details>
<summary><strong>Backend Diagnostic Helpers</strong></summary>

| Method | Description |
|---|---|
| `torch.backends.cuda.flash_sdp_enabled()` | Returns `True` if FlashAttention is currently available and enabled |
| `torch.backends.cuda.mem_efficient_sdp_enabled()` | Returns `True` if Memory-Efficient Attention is available |
| `torch.backends.cuda.cudnn_sdp_enabled()` | Returns `True` if cuDNN attention pathways are available |
| `torch.backends.cuda.math_sdp_enabled()` | Returns `True` if the C++ math fallback is enabled |

```python
import torch

# 1. Check compatibility of each backend before dispatching
print("FlashAttention available:      ", torch.backends.cuda.flash_sdp_enabled())
print("Mem-Efficient Attention avail: ", torch.backends.cuda.mem_efficient_sdp_enabled())
print("cuDNN Attention available:     ", torch.backends.cuda.cudnn_sdp_enabled())
print("Math (C++) fallback available: ", torch.backends.cuda.math_sdp_enabled())
```

</details>

</details>

---

## Distributed Parallelism & Model Sharding

<details>
<summary><strong>FSDP1 — FullyShardedDataParallel (Eager Mode)</strong></summary>

---

### `class torch.distributed.fsdp.FullyShardedDataParallel(module, sharding_strategy, cpu_offload, device_mesh, mixed_precision, ...)`

Wraps a model module to shard parameters, gradients, and optimizer states across a data-parallel device mesh using ZeRO Stage 3 paradigms.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `module` | `nn.Module` | *Required* | The target transformer layer or network block to be sharded |
| `sharding_strategy` | `ShardingStrategy` | `FULL_SHARD` | Sets partitioning depth: `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD` |
| `cpu_offload` | `CPUOffload / None` | `None` | Offloads sharded states to system RAM to bypass GPU memory limits |
| `device_mesh` | `DeviceMesh / None` | `None` | Maps multi-dimensional layouts, combining tensor and data parallelism |
| `mixed_precision` | `MixedPrecision / None` | `None` | Controls `param_dtype`, `reduce_dtype`, and `output_dtype` collectively |

> ℹ️ **Wrapping order matters:** Apply `FullyShardedDataParallel` **bottom-up**, wrapping leaf submodules first, then parent modules. This ensures distinct communication boundaries per transformer block.

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaForCausalLM, LlamaConfig
import functools

# 1. Initialize the process group
dist.init_process_group(backend="nccl")

# 2. Build a large transformer model on meta device
config = LlamaConfig()
with torch.device("meta"):
    model = LlamaForCausalLM(config)

# 3. Define auto-wrap policy targeting transformer decoder layers
auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={model.model.layers[0].__class__}
)

# 4. Configure mixed precision (bf16 compute, fp32 reduction)
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=torch.bfloat16,
)

# 5. Wrap model with FSDP1 (bottom-up via auto_wrap_policy)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap,
    device_id=torch.cuda.current_device(),
)

print(f"FSDP model wrapped on rank {dist.get_rank()}")
```

</details>

---

<details>
<summary><strong>FSDP2 — fully_shard (Compiler-Integrated)</strong></summary>

---

### `def torch.distributed._composable.fsdp.fully_shard(module, *, mesh, reshard_after_forward, mp_policy, offload_policy)`

Applies compiler-native FSDP2 sharding by converting model parameters to DTensor primitives sharded along dimension-0, enabling full `torch.compile` integration.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `module` | `nn.Module` | *Required* | The target transformer layer or network block to be sharded |
| `mesh` | `DeviceMesh` | *Required* | Device mesh defining the data-parallel device layout |
| `reshard_after_forward` | `bool` | `True` | Frees unsharded parameters after each forward pass to reclaim memory |
| `mp_policy` | `MixedPrecisionPolicy / None` | `None` | Direct `param_dtype`, `reduce_dtype`, `output_dtype` configuration |
| `offload_policy` | `OffloadPolicy / None` | `None` | Controls CPU offload behavior at the per-module level |

> ℹ️ **Key distinction from FSDP1:** FSDP2 uses DTensor-based sharding with no Python runtime hooks, making the full parameter lifecycle visible to `torch.compile` as standard graph nodes.

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import LlamaForCausalLM, LlamaConfig

# 1. Initialize distributed environment
dist.init_process_group(backend="nccl")

# 2. Build a 1-D data-parallel device mesh across all GPUs
mesh = init_device_mesh("cuda", (dist.get_world_size(),))

# 3. Instantiate the model
config = LlamaConfig()
model = LlamaForCausalLM(config).cuda()

# 4. Define mixed precision policy directly on FSDP2
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# 5. Apply FSDP2 bottom-up: wrap leaf layers first
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

# 6. Wrap the root module last
fully_shard(model, mesh=mesh, mp_policy=mp_policy)

# 7. Compile the sharded model — FSDP2 collectives are graph-visible
compiled_model = torch.compile(model)
print("FSDP2 model compiled and ready.")
```

</details>

---

<details>
<summary><strong>Tensor Parallelism — parallelize_module</strong></summary>

---

### `def torch.distributed.tensor.parallel.parallelize_module(module, device_mesh, parallelize_plan)`

Applies column-wise and row-wise tensor parallelism to attention and MLP projections by distributing weight tensors across a sub-device mesh using DTensor.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `module` | `nn.Module` | *Required* | The transformer block containing attention and MLP projections to split |
| `device_mesh` | `DeviceMesh` | *Required* | The sub-device mesh defining the tensor-parallel device layout |
| `parallelize_plan` | `dict[str, Placement]` | *Required* | Maps named submodules to `ColwiseParallel` or `RowwiseParallel` placements |

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from transformers import LlamaForCausalLM, LlamaConfig

# 1. Initialize process group
dist.init_process_group(backend="nccl")

# 2. Build a 2-D mesh: outer dim = data parallel, inner dim = tensor parallel
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh_2d["tp"]

# 3. Instantiate model
config = LlamaConfig()
model = LlamaForCausalLM(config).cuda()

# 4. Apply tensor parallelism to the first decoder layer
for layer in model.model.layers:
    parallelize_module(
        layer,
        device_mesh=tp_mesh,
        parallelize_plan={
            # Column-wise: splits output dimension across TP devices
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            # Row-wise: splits input dimension, performs all-reduce after
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj":    ColwiseParallel(),
            "mlp.up_proj":      ColwiseParallel(),
            "mlp.down_proj":    RowwiseParallel(),
        }
    )

print("Tensor parallelism applied to all decoder layers.")
```

</details>

---

## Graph Compilation & Backend Fusion

<details>
<summary><strong>torch.compile — Compiler-Driven Graph Optimization</strong></summary>

---

### `def torch.compile(model, *, fullgraph, dynamic, backend, mode)`

Transforms imperative Python model code into optimized machine-code execution graphs using TorchDynamo for capture, AOTAutograd for backward tracing, and TorchInductor for kernel generation.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `Callable` | *Required* | The PyTorch model or function to compile |
| `fullgraph` | `bool` | `False` | Requires the entire model to be captured in a single graph; raises on graph breaks |
| `dynamic` | `bool / None` | `None` | Enables dynamic shape tracing to avoid recompilation on varying sequence lengths |
| `backend` | `str` | `"inductor"` | Compilation backend; `"inductor"` generates fused CUDA kernels via TorchInductor |
| `mode` | `str / None` | `None` | Preset tuning mode: `"default"`, `"reduce-overhead"`, or `"max-autotune"` |

> ℹ️ **FSDP2 + compile:** Because FSDP2 represents collective communications as standard DTensor graph nodes, `torch.compile` can fuse collective scheduling with matrix multiplications, eliminating Python-hook-induced graph breaks present in FSDP1.

```python
import torch
from torch.distributed._composable.fsdp import fully_shard
from transformers import LlamaForCausalLM, LlamaConfig

# 1. Build and shard the model with FSDP2 (compiler-native)
config = LlamaConfig()
model = LlamaForCausalLM(config).cuda()

# 2. Compile the full model graph
#    "reduce-overhead" uses CUDA graphs to eliminate Python overhead per step
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead",
    fullgraph=False,   # Allow partial graph capture during warm-up
    dynamic=True,      # Handle varying sequence lengths without recompile
)

# 3. Run a compiled forward pass
input_ids = torch.randint(0, 32000, (1, 512), device="cuda")
with torch.no_grad():
    output = compiled_model(input_ids)

print("Compiled forward pass complete. Logits shape:", output.logits.shape)
```

</details>

---

## Telemetry, Logging & Post-Training Frameworks

<details>
<summary><strong>torch._logging.set_logs — Diagnostic Telemetry</strong></summary>

---

### `def torch._logging.set_logs(*, fsdp, dtensor, autotuning, graph_region_expansion, inductor_metrics)`

Configures structured diagnostic logging verbosity for PyTorch's distributed execution subsystems and compiler pipelines.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `fsdp` | `int / logging.Level` | `logging.WARNING` | FSDP logging detail level; use `logging.DEBUG` for all-gather/reduce-scatter traces |
| `dtensor` | `int / logging.Level` | `logging.WARNING` | DTensor tracing levels for distributed tensor operations |
| `autotuning` | `int / logging.Level` | `logging.WARNING` | Inductor kernel autotuning logs for hardware-specific kernel selection |
| `graph_region_expansion` | `int / logging.Level` | `logging.WARNING` | Graph compiler metrics for expansion and region boundary decisions |
| `inductor_metrics` | `int / logging.Level` | `logging.WARNING` | Estimated runtime tracking from the TorchInductor compilation pipeline |

```python
import torch
import logging

# 1. Enable verbose diagnostics for FSDP communication and DTensor operations
torch._logging.set_logs(
    fsdp=logging.DEBUG,               # Log all-gather / reduce-scatter events
    dtensor=logging.DEBUG,            # Log DTensor placement and resharding
    autotuning=logging.INFO,          # Log kernel selection decisions
    graph_region_expansion=logging.INFO,
    inductor_metrics=logging.INFO,    # Log estimated kernel runtimes
)

# 2. Run model forward pass — logs will print to stderr
model = torch.nn.Linear(128, 128).cuda()
compiled = torch.compile(model)
x = torch.randn(8, 128, device="cuda")
out = compiled(x)
print("Output shape:", out.shape)
```

</details>

---

<details>
<summary><strong>torchtune.generation.generate — Autoregressive Text Generation</strong></summary>

---

### `def torchtune.generation.generate(model, prompt, max_generated_tokens, pad_id, temperature, top_k, stop_tokens, rng, custom_generate_next_token)`

Executes an autoregressive token generation loop over a transformer model, applying temperature scaling and top-k sampling on the accelerator to minimize host synchronization.

$$\tilde{l}_i = \frac{l_i}{T}$$

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `nn.Module` | *Required* | The compiled or eager transformer model |
| `prompt` | `Tensor` | *Required* | Input sequence tensor of token IDs |
| `max_generated_tokens` | `int` | *Required* | Maximum number of new tokens to generate |
| `pad_id` | `int` | *Required* | Padding token index for batch alignment |
| `temperature` | `float` | `1.0` | Scales logit distribution entropy; lower values produce more deterministic output |
| `top_k` | `int / None` | `None` | Restricts sampling to the top-k highest probability tokens |
| `stop_tokens` | `list[int] / None` | `None` | Token IDs that terminate generation when produced |
| `rng` | `torch.Generator / None` | `None` | Random number generator for reproducible sampling |
| `custom_generate_next_token` | `Callable / None` | `None` | Optional compiled hook grouping KV-cache lookup, scaling, and sampling into a single graph |

```python
import torch
from torchtune.generation import generate
from torchtune.models.llama3 import llama3_8b

# 1. Load model and tokenizer
model = llama3_8b().cuda()
model.eval()

# 2. Encode prompt into token IDs
tokenizer_output = tokenizer.encode("Explain self-attention in transformers.")
prompt = torch.tensor(tokenizer_output, device="cuda").unsqueeze(0)

# 3. Define a compiled next-token function to minimize Python overhead per step
compiled_next_token = torch.compile(
    model.generate_next_token,
    fullgraph=True,
    mode="reduce-overhead"
)

# 4. Run autoregressive generation
rng = torch.Generator(device="cuda")
rng.manual_seed(42)

output_tokens, _ = generate(
    model=model,
    prompt=prompt,
    max_generated_tokens=256,
    pad_id=tokenizer.pad_id,
    temperature=0.8,
    top_k=50,
    stop_tokens=[tokenizer.eos_id],
    rng=rng,
    custom_generate_next_token=compiled_next_token,
)

# 5. Decode output tokens
output_text = tokenizer.decode(output_tokens[0].tolist())
print(output_text)
```

</details>

---

<details>
<summary><strong>torchtune.generation.sample — Token Sampling</strong></summary>

---

### `def torchtune.generation.sample(logits, temperature, top_k, q)`

Applies temperature scaling and top-k pruning to raw logits, then samples the next token index using a pre-drawn random tensor to avoid repeated host-device synchronization.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `logits` | `Tensor` | *Required* | Unnormalized predictions tensor of shape `(B, V)` from the model's final linear layer |
| `temperature` | `float` | *Required* | Distribution entropy scaling factor; values below 1.0 sharpen the distribution |
| `top_k` | `int / None` | `None` | Limits selection to the top `k` most probable tokens before sampling |
| `q` | `Tensor / None` | `None` | Pre-sampled random tensor for softmax sampling; avoids mid-loop CPU synchronization |

```python
import torch
from torchtune.generation import sample

# 1. Simulate raw logits from a model forward pass
vocab_size = 32000
logits = torch.randn(1, vocab_size, device="cuda")

# 2. Pre-draw the random tensor on the accelerator (avoids CPU sync)
q = torch.empty_like(logits).exponential_(1.0)

# 3. Sample the next token with temperature and top-k filtering
next_token = sample(
    logits=logits,
    temperature=0.7,
    top_k=50,
    q=q,
)

print("Sampled token ID:", next_token.item())
```

</details>

---

---

# Client-Service Integration SDK

> High-level SDK layer managing API network boundaries, authentication, streaming, batching, extended reasoning, and managed agent workflows.

---

## Client Lifecycle & Authentication

<details>
<summary><strong>Anthropic / AsyncAnthropic — Client Initialization</strong></summary>

---

### `class anthropic.Anthropic(api_key, auth_token, credentials, config, profile, http_client, max_retries, base_url)`

Initializes a synchronous Anthropic API client, resolving credentials through a strict parameter precedence chain.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `api_key` | `str / None` | `None` | Static API key; takes precedence over env auto-discovery; reads `ANTHROPIC_API_KEY` if absent |
| `auth_token` | `str / Callable` | `None` | Bearer token; can be a callable for per-request automated token rotation |
| `credentials` | `AccessTokenProvider` | `None` | Explicit token provider; mutually exclusive with `config` and `profile` |
| `config` | `Mapping[str, Any]` | `None` | Dictionary of explicit configuration parameters |
| `profile` | `str` | `None` | Reference profile path on disk for profile-based auto-discovery |
| `http_client` | `httpx.Client` | `None` | Customizable HTTP transport layer for injecting proxies or adjusting timeouts |
| `max_retries` | `int` | `2` | Maximum connection retry attempts on transient errors |
| `base_url` | `str` | `None` | Target gateway URL for routing requests through internal proxy systems |

> ⚠️ **Precedence rule:** If both a static key (`api_key` / `auth_token`) and a dynamic provider (`credentials`) are passed simultaneously, the static parameter wins and a `_warn_env_shadow` warning is logged.

```python
import anthropic
import httpx

# 1. Default initialization — reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# 2. Explicit static API key
client = anthropic.Anthropic(api_key="sk-ant-...")

# 3. Dynamic token rotation via callable (enterprise short-lived tokens)
def rotate_token() -> str:
    """Called per-request; fetches a fresh bearer token from your auth service."""
    return fetch_fresh_token_from_auth_server()

client = anthropic.Anthropic(auth_token=rotate_token)

# 4. Route through a local proxy with a custom httpx transport
proxy_client = httpx.Client(proxies={"https://": "http://localhost:8080"})
client = anthropic.Anthropic(
    api_key="sk-ant-...",
    http_client=proxy_client,
    max_retries=3,
)

# 5. Override the API gateway for self-hosted deployments
client = anthropic.Anthropic(
    api_key="sk-ant-...",
    base_url="https://internal-proxy.company.com/anthropic",
)
```

---

### `class anthropic.AsyncAnthropic(...)`

Asynchronous variant of the Anthropic client; accepts identical parameters — all methods must be awaited.

```python
import asyncio
import anthropic

async def main():
    # 1. Initialize the async client
    client = anthropic.AsyncAnthropic(api_key="sk-ant-...")

    # 2. Await the message creation call
    message = await client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": "Explain DTensor in one paragraph."}]
    )

    # 3. Access the text response
    print(message.content[0].text)

asyncio.run(main())
```

</details>

---

## Messages Interface & Streaming

<details>
<summary><strong>client.messages.create — Synchronous Completion</strong></summary>

---

### `def client.messages.create(model, messages, max_tokens, system, temperature, tools, tool_choice, thinking, stream, ...)`

Sends a structured conversation payload to the model service and returns a completion response.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model identifier (e.g. `"claude-opus-4-6"`) |
| `messages` | `list[dict]` | *Required* | Alternating `user` / `assistant` turns; consecutive same-role messages are auto-merged |
| `max_tokens` | `int` | *Required* | Hard cap on output tokens |
| `system` | `str` | `None` | System-level instructions prepended to the conversation |
| `temperature` | `float` | `1.0` | Output randomness; lower values produce more deterministic responses |
| `tools` | `list[Tool]` | `None` | Structured tool definitions Claude can invoke |
| `tool_choice` | `dict` | `"auto"` | Tool-calling mode: `"auto"`, `"any"`, `"none"`, or `{"type": "tool", "name": "..."}` |
| `thinking` | `dict` | `None` | Extended thinking config: `{"type": "enabled", "budget_tokens": N}` or `{"type": "adaptive"}` |
| `stream` | `bool` | `False` | When `True`, returns a raw async SSE iterator instead of a complete response |

```python
import anthropic

client = anthropic.Anthropic()

# 1. Basic single-turn completion
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is tensor parallelism?"}
    ]
)
print(response.content[0].text)

# 2. Multi-turn conversation (user → assistant → user)
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=512,
    system="You are a distributed systems expert.",
    messages=[
        {"role": "user",      "content": "Explain FSDP."},
        {"role": "assistant", "content": "FSDP shards model parameters across GPUs..."},
        {"role": "user",      "content": "How does FSDP2 differ?"},
    ]
)
print(response.content[0].text)

# 3. Tool use — force a specific tool call
tools = [{
    "name": "get_gpu_utilization",
    "description": "Returns current GPU memory usage.",
    "input_schema": {
        "type": "object",
        "properties": {"device_id": {"type": "integer"}},
        "required": ["device_id"]
    }
}]
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=256,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_gpu_utilization"},
    messages=[{"role": "user", "content": "Check GPU 0 memory usage."}]
)
print(response.content)
```

</details>

---

<details>
<summary><strong>Streaming — messages.create(stream=True) vs messages.stream()</strong></summary>

---

### `def client.messages.create(..., stream=True)` — Raw SSE Iterator

Returns a raw `AsyncIterable[Event]` of Server-Sent Events with minimal memory overhead; requires manual text assembly.

| Approach | Underlying Type | Memory | Behavior |
|---|---|:---:|---|
| `messages.create(stream=True)` | `AsyncIterable[Event]` | Minimal | Raw SSE iterator; manual token assembly required |
| `messages.stream(...)` | `BetaMessageStreamManager` | Intermediate | Context manager; auto-accumulates text deltas and token events |

```python
import anthropic

client = anthropic.Anthropic()

# ── Approach A: Raw SSE stream (manual assembly) ──────────────────────────────

# 1. Request a raw stream
stream = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Describe FlashAttention."}],
    stream=True,
)

# 2. Iterate and assemble text deltas manually
full_text = ""
for event in stream:
    if event.type == "content_block_delta" and hasattr(event.delta, "text"):
        full_text += event.delta.text
        print(event.delta.text, end="", flush=True)
    elif event.type == "message_stop":
        break

print("\n--- Stream complete ---")
```

---

### `def client.messages.stream(...)` — Managed Context Manager

Wraps the raw SSE stream in a `BetaMessageStreamManager`, automatically accumulating text deltas and exposing a clean `text_stream` iterator.

```python
import anthropic

client = anthropic.Anthropic()

# 1. Use the stream context manager (recommended)
with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Describe Memory-Efficient Attention."}]
) as stream:
    # 2. Iterate over pre-assembled text chunks
    for text_chunk in stream.text_stream:
        print(text_chunk, end="", flush=True)

    # 3. Retrieve the fully accumulated final message after streaming
    final_message = stream.get_final_message()
    print(f"\n\nTotal tokens used: {final_message.usage.output_tokens}")
```

> ⚠️ **Vertex AI / Bedrock:** When using cloud-platform wrappers with the beta messages API, the wrapper may bypass `BetaMessageStreamManager` and return a raw stream. In these cases, use Approach A (raw SSE) and assemble deltas manually.

</details>

---

## Batch Operations & Async Queue Processing

<details>
<summary><strong>client.messages.batches — Async Batch Pipeline</strong></summary>

---

### `def client.messages.batches.create(requests)`

Submits a batch of up to 100,000 inference requests to an asynchronous cloud queue for processing at 50% cost reduction.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `requests` | `list[BatchRequest]` | *Required* | List of request objects; each must include `custom_id` plus standard Messages API params |

> **Resource limits:** Max 100,000 requests or 256 MB per batch. Results guaranteed within 24 hours. Output accessible for 29 days post-completion.

```python
import anthropic

client = anthropic.Anthropic()

# ── Batch Lifecycle ────────────────────────────────────────────────────────────

# 1. Create a batch of requests
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"req-{i:04d}",
            "params": {
                "model": "claude-opus-4-6",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": f"Summarize topic #{i}."}]
            }
        }
        for i in range(500)
    ]
)
print(f"Batch created: {batch.id}  |  Status: {batch.processing_status}")

# 2. Check processing status
status = client.messages.batches.retrieve(batch.id)
print(f"Status: {status.processing_status}")

# 3. List all active and completed batches
for b in client.messages.batches.list():
    print(f"  {b.id}  {b.processing_status}")

# 4. Stream results when processing is complete
if status.processing_status == "ended":
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            print(f"[{result.custom_id}] {result.result.message.content[0].text[:80]}")
        else:
            print(f"[{result.custom_id}] Error: {result.result.error.type}")

# 5. Cancel a pending batch (before processing completes)
client.messages.batches.delete(batch.id)

# 6. Archive completed batch results from the remote workspace
client.messages.batches.archive(batch.id)
```

</details>

---

## Extended Thinking & Token Budgets

<details>
<summary><strong>thinking parameter — Extended Reasoning Configuration</strong></summary>

---

### Extended Thinking Configuration via `thinking` parameter in `messages.create`

Enables internal multi-step reasoning before the model produces its final response, returning `type: "thinking"` blocks followed by `type: "text"` blocks.

| Configuration Key | Type | Default | Description |
|---|---|:---:|---|
| `type` | `str` | *Required* | `"enabled"` for manual budget, `"adaptive"` for dynamic allocation |
| `budget_tokens` | `int` | *Required (if enabled)* | Fixed token budget for internal planning steps; billed at standard output rates |

| `effort` Level | Target Behavior |
|---|---|
| `max` | Maximum possible token budget; highest accuracy on complex tasks |
| `high` | Equivalent to omitting the parameter; thorough answers |
| `medium` | Balances accuracy and latency by reducing the thinking budget |
| `low` | Minimal thinking steps; optimizes for low latency and cost |

> ⚠️ **Tool-use restriction:** Extended thinking is **only compatible** with `tool_choice: {"type": "auto"}` or `tool_choice: {"type": "none"}`. Using `"any"` or explicit tool selection will cause the API to reject the request.

> ℹ️ **Omitted mode (Claude Opus 4.8+):** When `display` defaults to `"omitted"`, the planning text is replaced with an empty string, but an encrypted `signature` field is included. You **must** return this signature unmodified in subsequent turns to maintain reasoning continuity.

```python
import anthropic

client = anthropic.Anthropic()

# 1. Manual thinking — fixed token budget
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Design a distributed training scheduler for a 70B LLM."}]
)

# 2. Inspect response blocks
for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking — {len(block.thinking)} chars]\n{block.thinking[:200]}...\n")
    elif block.type == "text":
        print(f"[Final Answer]\n{block.text}")

# 3. Adaptive thinking — model adjusts budget based on prompt complexity
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=8000,
    thinking={"type": "adaptive"},
    messages=[{"role": "user", "content": "What is 2 + 2?"}]  # Simple: minimal budget used
)
print(response.content[0].text)

# 4. Multi-turn with omitted thinking — must return signature to preserve continuity
first_response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=[{"role": "user", "content": "Plan a sharding strategy for a 405B model."}]
)

# 5. Append assistant response (including empty thinking block with signature) to history
conversation = [
    {"role": "user", "content": "Plan a sharding strategy for a 405B model."},
    {"role": "assistant", "content": first_response.content},  # Includes signature block
    {"role": "user",      "content": "Now estimate the required interconnect bandwidth."},
]

followup = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=conversation,
)
print(followup.content[-1].text)
```

</details>

---

<details>
<summary><strong>client.messages.count_tokens — Pre-flight Token Validation</strong></summary>

---

### `def client.messages.count_tokens(model, messages, system, tools)`

Estimates input token usage without generating any output — used for cost validation and dynamic routing under Zero Data Retention (ZDR) agreements.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model identifier used for tokenization |
| `messages` | `list[dict]` | *Required* | Conversation history to count |
| `system` | `str` | `None` | System prompt text to include in the count |
| `tools` | `list[Tool]` | `None` | Tool definitions to include in the count |

> ℹ️ **Reasoning models:** Previous thinking steps are excluded from input counts. Thinking blocks in the **current** turn are included, enabling accurate usage tracking.

```python
import anthropic

client = anthropic.Anthropic()

# 1. Define the request payload
messages = [
    {"role": "user", "content": "Explain FSDP2 and how it differs from FSDP1."}
]
system = "You are an expert in distributed deep learning."

# 2. Count tokens before dispatching (no tokens generated, no cost)
token_count = client.messages.count_tokens(
    model="claude-opus-4-6",
    messages=messages,
    system=system,
)
print(f"Estimated input tokens: {token_count.input_tokens}")

# 3. Conditionally route based on token count
CONTEXT_LIMIT = 100_000
if token_count.input_tokens < CONTEXT_LIMIT:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    print(response.content[0].text)
else:
    print("Payload exceeds context limit — truncate or chunk the input.")
```

</details>

---

## Functional Tool Execution & Agent Control

<details>
<summary><strong>@beta_tool — Automatic JSON Schema Generation from Function Signatures</strong></summary>

---

### `@beta_tool` decorator on `def function_name(...)`

Automatically generates a JSON schema tool definition from a Python function's type annotations and docstring, eliminating manual schema authoring.

```python
import anthropic
from anthropic import beta_tool

client = anthropic.Anthropic()

# 1. Decorate a typed Python function — schema is auto-generated
@beta_tool
def get_gpu_memory(device_id: int, unit: str = "GB") -> dict:
    """
    Returns the current memory usage for a specified GPU device.

    Args:
        device_id: The CUDA device index (e.g., 0, 1, 2).
        unit: Memory unit to return — 'GB', 'MB', or 'bytes'.
    """
    import torch
    allocated = torch.cuda.memory_allocated(device_id)
    if unit == "GB":
        return {"device": device_id, "allocated": allocated / 1e9, "unit": "GB"}
    elif unit == "MB":
        return {"device": device_id, "allocated": allocated / 1e6, "unit": "MB"}
    return {"device": device_id, "allocated": allocated, "unit": "bytes"}

# 2. Inspect the auto-generated JSON schema
print(get_gpu_memory.tool_definition())

# 3. Pass the schema directly into a messages call
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=512,
    tools=[get_gpu_memory.tool_definition()],
    messages=[{"role": "user", "content": "Check memory usage on GPU 0 in GB."}]
)

# 4. Handle the tool call response
for block in response.content:
    if block.type == "tool_use":
        result = get_gpu_memory(**block.input)
        print(f"Tool result: {result}")
```

</details>

---

<details>
<summary><strong>disableParallelToolUse — Sequential Tool Execution Control</strong></summary>

---

### `tool_choice={"type": "auto", "disable_parallel_tool_use": True}` in `messages.create`

Forces the model to execute tools one at a time in sequence, preventing parallel tool invocation loops and controlling API cost accumulation.

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "read_file",
        "description": "Read a file from the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    }
]

# 1. Force sequential tool use (one tool call per turn)
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    messages=[{"role": "user", "content": "Read config.yaml, then update the batch_size to 64."}]
)

# 2. Process the single tool call returned per turn
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool called: {block.name}  |  Input: {block.input}")
```

</details>

---

<details>
<summary><strong>Managed Agent Control Plane — Agents, Environments & Sessions</strong></summary>

---

### `def client.beta.agents.create(model, system, tools, name)`

Creates a persistent managed agent definition with a bound model, system instructions, and an available toolset.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model identifier for the agent |
| `system` | `str` | `None` | System instructions defining the agent's role and behavior |
| `tools` | `list[Tool]` | `None` | Available tool definitions the agent can invoke |
| `name` | `str` | `None` | Human-readable agent name for identification |

---

### `def client.beta.environments.create(network_access, ...)`

Establishes an isolated, secure computing workspace with configurable network access policies for agent execution.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `network_access` | `str` | `"none"` | Network policy: `"none"`, `"restricted"`, or `"full"` |

---

### `def client.beta.sessions.create(agent_id, environment_id)`

Links a managed agent to an execution environment and initiates a tracked run session.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `agent_id` | `str` | *Required* | ID of the agent created via `client.beta.agents.create` |
| `environment_id` | `str` | *Required* | ID of the environment created via `client.beta.environments.create` |

---

### `def client.beta.files.upload(file_path, mime_type)`

Uploads a resource file and mounts it directly to the agent's workspace.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file_path` | `str` | *Required* | Local path to the file to upload (CSV, code, documents, etc.) |
| `mime_type` | `str` | `"application/octet-stream"` | MIME type of the uploaded file |

```python
import anthropic

client = anthropic.Anthropic()

# 1. Upload a reference dataset to the agent workspace
file_response = client.beta.files.upload(
    file_path="training_metrics.csv",
    mime_type="text/csv",
)
print(f"Uploaded file ID: {file_response.id}")

# 2. Define tools for the agent
tools = [{
    "name": "analyze_csv",
    "description": "Analyze training metrics from a mounted CSV file.",
    "input_schema": {
        "type": "object",
        "properties": {"file_id": {"type": "string"}},
        "required": ["file_id"]
    }
}]

# 3. Create a managed agent with system instructions
agent = client.beta.agents.create(
    model="claude-opus-4-6",
    name="MLOps Analyst",
    system="You are an expert in distributed training. Analyze metrics and identify bottlenecks.",
    tools=tools,
)
print(f"Agent created: {agent.id}")

# 4. Create an isolated execution environment
environment = client.beta.environments.create(network_access="none")
print(f"Environment created: {environment.id}")

# 5. Link agent to environment and start a session
session = client.beta.sessions.create(
    agent_id=agent.id,
    environment_id=environment.id,
)
print(f"Session started: {session.id}  |  Status: {session.status}")
```

</details>

---

---

*Reference compiled from PyTorch SDK documentation and Anthropic Client-Service SDK architecture.*  
*For live API specs see: [PyTorch Docs](https://pytorch.org/docs) · [Anthropic Docs](https://docs.anthropic.com)*