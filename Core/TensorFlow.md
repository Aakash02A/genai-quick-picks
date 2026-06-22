# TensorFlow GenAI SDK

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Part I — TensorFlow & Keras Ecosystem](#part-i--tensorflow--keras-ecosystem)
  - [Autoregressive Pretrained Architectures](#autoregressive-pretrained-architectures)
  - [Sampling Mechanics](#sampling-mechanics)
  - [Custom Neural Primitives](#custom-neural-primitives)
  - [Gradient & Optimization Primitives](#gradient--optimization-primitives)
  - [Generative Adversarial Networks](#generative-adversarial-networks)
- [Part II — Remote API-Driven SDK](#part-ii--remote-api-driven-sdk)
  - [Responses & Conversations Interface](#responses--conversations-interface)
  - [Realtime Streaming API](#realtime-streaming-api)
  - [Large-Scale Ingestion & Fine-Tuning](#large-scale-ingestion--fine-tuning)
  - [Video Synthesis API](#video-synthesis-api)
  - [Agentic Orchestration](#agentic-orchestration)
- [Part III — Hybrid Pipeline Architecture](#part-iii--hybrid-pipeline-architecture)

---

## Architecture Overview

Modern enterprise generative AI pipelines combine two complementary paradigms:

| Paradigm | Toolchain | Best For |
|---|---|---|
| **Local SDK** | TensorFlow + Keras | Custom training, on-premise security, GANs, VAEs |
| **Cloud API** | OpenAI Responses / Agents SDK | Reasoning, multi-agent orchestration, low-latency inference |

```
+-------------------------------------------------------------+
|              Local TensorFlow / Keras Pipeline              |
|   Raw Sensors ---> [ tf.keras.layers.MultiHeadAttention ]   |
|                                |                            |
|                                v                            |
|                    [ tf.keras.Model.predict() ]             |
+-------------------------------------------------------------+
                                 |
                                 | (Local Feature Vector)
                                 v
+-------------------------------------------------------------+
|                API-Driven Orchestration Layer               |
|   Vector Data ---> [ client.files.create(purpose="RAG") ]   |
|                                |                            |
|                                v                            |
|                    [ client.responses.create() ]            |
|                                |                            |
|                                v                            |
|                    [ openai-agents Execution ]              |
+-------------------------------------------------------------+
```

---

## Part I — TensorFlow & Keras Ecosystem

<details>
<summary><strong>Autoregressive Pretrained Architectures</strong></summary>

&nbsp;

> Load and run pretrained causal language and multimodal models from the KerasHub model hub.

---

### `class keras_hub.models.GPT2Tokenizer(vocabulary, merges, dtype)`

Converts raw text into structured integer token indices using byte-pair encoding (BPE).

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `vocabulary` | `dict` | *Required* | BPE vocabulary mapping tokens to integer IDs. |
| `merges` | `list` | *Required* | Ordered list of BPE merge rules applied during tokenization. |
| `dtype` | `str` | `"int32"` | Output tensor dtype for the token index array. |

---

### `class keras_hub.models.GPT2CausalLMPreprocessor(sequence_length, preprocessor)`

Manages sequence tokenization, end-of-turn token insertion, and input padding.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `sequence_length` | `int` | *Required* | Maximum token length; sequences are padded or truncated to this size. |
| `preprocessor` | `GPT2Tokenizer` | *Required* | Tokenizer instance used for encoding raw text inputs. |

---

### `class keras_hub.models.GPT2Backbone(vocabulary_size, num_layers, num_heads, hidden_dim)`

Houses stacked transformer decoder layers that apply self-attention over input token vectors.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `vocabulary_size` | `int` | *Required* | Total number of tokens in the model vocabulary. |
| `num_layers` | `int` | *Required* | Number of stacked transformer decoder blocks. |
| `num_heads` | `int` | *Required* | Number of parallel self-attention heads per layer. |
| `hidden_dim` | `int` | *Required* | Dimensionality of each token's hidden representation. |

---

### `class keras_hub.models.GemmaCausalLM(backbone, preprocessor, dtype)`

Orchestrates backbone and preprocessor to map decoder outputs into log-probability distributions over the full vocabulary.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `backbone` | `GPT2Backbone` | *Required* | Transformer backbone network to route token inputs through. |
| `preprocessor` | `GPT2CausalLMPreprocessor` | *Required* | Preprocessor that tokenizes and pads raw input text. |
| `dtype` | `str` | `"float32"` | Compute precision; use `"bfloat16"` for memory efficiency. |

#### `.from_preset(preset_name, dtype)`

Class method that loads pre-compiled weights and tokenizer vocabularies from remote repositories.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `preset_name` | `str` | *Required* | Identifier of the remote preset (e.g., `"gemma_2b_en"`). |
| `dtype` | `str` | `"float32"` | Compute precision override for the loaded model. |

#### `.generate(inputs, max_length)`

Executes continuous autoregressive token generation over a prompt sequence.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `inputs` | `str \| tf.Tensor` | *Required* | Seed text or token tensor to condition generation on. |
| `max_length` | `int` | *Required* | Maximum number of tokens to produce, including the prompt. |

```python
import os

# 1. Configure JAX backend with full memory allocation
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_hub

# 2. Load a production-grade preset with bfloat16 precision
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_2b_en",
    dtype="bfloat16"
)

# 3. Run autoregressive generation from a seed prompt
generated_text = gemma_lm.generate(
    "The architectural goal of multi-backend design is",
    max_length=64
)

# 4. Print the completed sequence
print(generated_text)
```

---

### Multimodal Execution — `PaliGemmaCausalLM`

Accepts joint tensor inputs containing both raw pixel values and text prompts, routing them through a visual encoder and transformer backbone.

| Execution Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `inputs["images"]` | `np.ndarray \| tf.Tensor` | *Required* | Raw image matrices, resized and normalized by the image preprocessor. |
| `inputs["prompts"]` | `str \| tf.Tensor` | *Required* | Textual prompt containing target questions and formatting templates. |
| `stop_token_ids` | `list[int]` | `None` | List of token IDs that halt generation upon prediction. |

</details>

---

<details>
<summary><strong>Sampling Mechanics</strong></summary>

&nbsp;

> Control token selection strategy at inference time. Samplers are attached via `.compile()` and can be swapped without modifying backbone weights.

---

### `class keras_hub.samplers.GreedySampler()`

Selects the single highest-probability token at every step, producing fully deterministic output.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| — | — | — | No configurable parameters. |

---

### `class keras_hub.samplers.BeamSampler(num_beams)`

Maintains parallel sequence paths and selects the path with the highest cumulative log-probability.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `num_beams` | `int` | *Required* | Number of candidate sequence paths to evaluate simultaneously. |

---

### `class keras_hub.samplers.TopKSampler(k)`

Restricts token selection to the top `k` candidates, redistributing probability mass among them before sampling.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `k` | `int` | *Required* | Number of top-ranked vocabulary candidates to sample from. |

---

### `class keras_hub.samplers.TopPSampler(p)`

Dynamically selects the smallest candidate set whose cumulative probability meets threshold `p`.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `p` | `float` | *Required* | Cumulative probability ceiling (e.g., `0.9`) for nucleus sampling. |

---

### `class keras_hub.samplers.ContrastiveSampler(k, alpha)`

Penalizes repetitive token selections using a contrastive degeneration objective to improve output diversity.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `k` | `int` | *Required* | Candidate pool size for contrastive evaluation. |
| `alpha` | `float` | *Required* | Penalty weight applied to repeated or degenerate token choices. |

```python
import keras_hub

# 1. Load base causal language model
model = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

# 2. Compile with Top-P (nucleus) sampler for diverse output
model.compile(sampler=keras_hub.samplers.TopPSampler(p=0.9))

# 3. Run inference — sampler applies without touching backbone weights
output = model.generate("The future of generative AI is", max_length=80)
print(output)

# 4. Hot-swap to Beam Search for higher-quality structured output
model.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=5))
output_beam = model.generate("The future of generative AI is", max_length=80)
print(output_beam)
```

</details>

---

<details>
<summary><strong>Custom Neural Primitives</strong></summary>

&nbsp;

> Low-level Keras layers for building custom transformer decoders, GAN generators, and VAE architectures from scratch.

---

### `class tf.keras.layers.MultiHeadAttention(num_heads, key_dim, value_dim, dropout, use_bias, output_shape, attention_axes)`

Computes scaled dot-product attention across multiple parallel projection heads, enabling the model to focus on different sequence positions simultaneously.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Initialization Parameters**

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `num_heads` | `int` | *Required* | Number of attention heads for parallel query/key/value projections. |
| `key_dim` | `int` | *Required* | Dimensionality of each head's query and key projection space. |
| `value_dim` | `int` | `None` | Head size for value projections; defaults to `key_dim` if unset. |
| `dropout` | `float` | `0.0` | Dropout rate applied to computed attention weight matrices. |
| `use_bias` | `bool` | `True` | Whether to add trainable bias vectors to Q, K, V, and output projections. |
| `output_shape` | `tuple` | `None` | Expected output shape, excluding batch and sequence dimensions. |
| `attention_axes` | `tuple` | `None` | Specific spatial or sequence axes over which attention is computed. |

**Call Arguments**

| Argument | Shape | Description |
|:---|:---|:---|
| `query` | `(B, T, dim)` | Query representations evaluated against all key positions. |
| `value` | `(B, S, dim)` | Value representations aggregated by attention weights. |
| `key` | `(B, S, dim)` | Key representations indexed against queries; defaults to `value`. |
| `attention_mask` | `(B, T, S)` | Boolean mask: `1` permits attention, `0` blocks interaction. |
| `return_attention_scores` | `bool` | If `True`, returns raw attention score matrices alongside output. |
| `training` | `bool` | Enables training-specific behaviors such as attention dropout. |
| `use_causal_mask` | `bool` | Enforces a lower-triangular mask to block future token positions. |

```python
import tensorflow as tf

# 1. Initialize multi-head attention with 8 heads, key dimension 64
mha = tf.keras.layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64,
    dropout=0.1
)

# 2. Create dummy sequence tensors (batch=2, seq_len=32, dim=512)
query = tf.random.normal((2, 32, 512))
value = tf.random.normal((2, 32, 512))

# 3. Forward pass with causal masking enforced (decoder-style)
output, scores = mha(
    query=query,
    value=value,
    use_causal_mask=True,
    return_attention_scores=True,
    training=True
)

print("Output shape:", output.shape)   # (2, 32, 512)
print("Scores shape:", scores.shape)   # (2, 8, 32, 32)
```

---

### `class tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias)`

Performs transposed convolution (spatial upsampling) to project compressed latent tensors into expanded spatial feature maps for generative vision models.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `filters` | `int` | *Required* | Number of output channels in the upsampled feature map. |
| `kernel_size` | `int \| tuple` | *Required* | Height and width of the transposed convolution window. |
| `strides` | `int \| tuple` | `(1, 1)` | Upsampling stride; `(2, 2)` doubles both spatial dimensions. |
| `padding` | `str` | `"valid"` | `"same"` for proportional upsampling; `"valid"` for no padding. |
| `output_padding` | `int \| tuple` | `None` | Explicit padding added along output height and width. |
| `data_format` | `str` | `None` | Tensor layout: `"channels_last"` or `"channels_first"`. |
| `dilation_rate` | `int \| tuple` | `(1, 1)` | Dilation rate; values `> 1` are incompatible with `strides > 1`. |
| `activation` | `callable` | `None` | Non-linear activation applied post-convolution (e.g., `"relu"`). |
| `use_bias` | `bool` | `True` | Whether to append a trainable bias vector to upsampled output. |

```python
import tensorflow as tf

# 1. Define a simple VAE decoder using successive transposed convolutions
decoder = tf.keras.Sequential([
    # 2. Project latent vector: 8x8x1024 → 16x16x512
    tf.keras.layers.Conv2DTranspose(
        filters=512, kernel_size=4, strides=2,
        padding="same", activation="relu"
    ),
    # 3. Upsample: 16x16x512 → 32x32x256
    tf.keras.layers.Conv2DTranspose(
        filters=256, kernel_size=4, strides=2,
        padding="same", activation="relu"
    ),
    # 4. Final projection to RGB image: → 64x64x3
    tf.keras.layers.Conv2DTranspose(
        filters=3, kernel_size=4, strides=2,
        padding="same", activation="tanh"
    ),
])

# 5. Pass a batch of latent tensors through the decoder
latent = tf.random.normal((4, 8, 8, 1024))
image_output = decoder(latent)
print("Decoded image shape:", image_output.shape)  # (4, 64, 64, 3)
```

</details>

---

<details>
<summary><strong>Gradient & Optimization Primitives</strong></summary>

&nbsp;

> Context manager for recording operations on tensors and computing exact partial derivatives via reverse-mode automatic differentiation.

---

### `class tf.GradientTape(persistent, watch_accessed_variables)`

Records tensor operations to compute gradients via reverse-mode automatic differentiation, enabling custom training loops.

$$\nabla_{\theta} \mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial w_1}, \frac{\partial \mathcal{L}}{\partial w_2}, \dots, \frac{\partial \mathcal{L}}{\partial w_n} \right]^T$$

**Constructor Parameters**

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `persistent` | `bool` | `False` | If `True`, allows multiple `.gradient()` calls on the same tape instance. |
| `watch_accessed_variables` | `bool` | `True` | If `False`, disables auto-tracking; requires manual `.watch()` calls. |

**Methods**

| Method | Signature | Description |
|:---|:---|:---|
| `.watch()` | `watch(tensor)` | Registers a constant tensor or non-trainable variable for gradient tracking. |
| `.gradient()` | `gradient(target, sources)` | Computes partial derivatives of `target` with respect to each source in `sources`. |
| `.jacobian()` | `jacobian(target, sources)` | Computes the full Jacobian matrix for vector-valued target outputs. |
| `.watched_variables()` | `watched_variables()` | Returns a list of all variables currently monitored by the tape. |
| `.stop_recording()` | `stop_recording()` | Temporarily suspends operation recording to conserve memory. |

```python
import tensorflow as tf

# 1. Define a simple model and optimizer
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()

x = tf.random.normal((32, 5))
y = tf.random.normal((32, 10))

# 2. Open a gradient tape context — operations are recorded automatically
with tf.GradientTape() as tape:
    # 3. Forward pass: compute predictions and scalar loss
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)

# 4. Compute gradients of loss w.r.t. all trainable weights
gradients = tape.gradient(loss, model.trainable_variables)

# 5. Apply gradient update to model parameters
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print(f"Loss: {loss.numpy():.4f}")
```

</details>

---

<details>
<summary><strong>Generative Adversarial Networks</strong></summary>

&nbsp;

> Custom Keras model implementing the GAN minimax objective with alternating discriminator and generator optimization via persistent gradient tapes.

---

### `class GenerativeAdversarialNetwork(generator, discriminator)`

Encapsulates the GAN minimax objective and implements alternating optimization of generator and discriminator using persistent gradient tapes.

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$$

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `generator` | `tf.keras.Model` | *Required* | Neural network mapping latent vectors to synthetic data samples. |
| `discriminator` | `tf.keras.Model` | *Required* | Neural network classifying inputs as real or generated. |

#### `.compile(gen_optimizer, disc_optimizer, loss_fn)`

Registers separate optimizers and a shared loss function for each sub-network.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `gen_optimizer` | `tf.keras.optimizers.Optimizer` | *Required* | Optimizer used for generator weight updates. |
| `disc_optimizer` | `tf.keras.optimizers.Optimizer` | *Required* | Optimizer used for discriminator weight updates. |
| `loss_fn` | `callable` | *Required* | Binary cross-entropy or equivalent loss for real/fake classification. |

#### `@tf.function .train_step(real_images)`

Executes one alternating optimization step: updates the discriminator on real and fake samples, then updates the generator via the fooling objective.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `real_images` | `tf.Tensor` | *Required* | Batch of real training images drawn from the data distribution. |

```python
import tensorflow as tf

# 1. Define lightweight generator and discriminator networks
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(100,)),
    tf.keras.layers.Dense(784, activation="tanh")
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 2. Instantiate the GAN wrapper model
class GenerativeAdversarialNetwork(tf.keras.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, 100))

        # 3. Record all forward ops for both sub-networks in one persistent tape
        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            predictions_real = self.discriminator(real_images, training=True)
            predictions_fake = self.discriminator(generated_images, training=True)

            loss_disc_real = self.loss_fn(tf.ones_like(predictions_real), predictions_real)
            loss_disc_fake = self.loss_fn(tf.zeros_like(predictions_fake), predictions_fake)
            loss_discriminator = loss_disc_real + loss_disc_fake
            loss_generator = self.loss_fn(tf.ones_like(predictions_fake), predictions_fake)

        # 4. Update discriminator weights
        gradients_disc = tape.gradient(loss_discriminator, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_disc, self.discriminator.trainable_variables))

        # 5. Update generator weights
        gradients_gen = tape.gradient(loss_generator, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_gen, self.generator.trainable_variables))

        # 6. Explicitly free persistent tape memory
        del tape
        return {"disc_loss": loss_discriminator, "gen_loss": loss_generator}

# 7. Compile and train
gan = GenerativeAdversarialNetwork(generator, discriminator)
gan.compile(
    gen_optimizer=tf.keras.optimizers.Adam(1e-4),
    disc_optimizer=tf.keras.optimizers.Adam(1e-4),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

real_data = tf.random.normal((1000, 784))
dataset = tf.data.Dataset.from_tensor_slices(real_data).batch(32)
gan.fit(dataset, epochs=10)
```

</details>

---

## Part II — Remote API-Driven SDK

<details>
<summary><strong>Responses & Conversations Interface</strong></summary>

&nbsp;

> Stateful, server-side text generation supporting multi-turn reasoning, structured output schema mapping, and persistent conversation threads.

---

### `def client.responses.create(model, input, instructions, store, previous_response_id, text, reasoning, max_output_tokens)`

Creates a stateful generation response with optional multi-turn conversation linking and structured JSON output enforcement.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | Model deployment target identifier (e.g., `"gpt-5.5"`). |
| `input` | `str \| list` | *Required* | Prompt text, image payload, or conversation history array. |
| `instructions` | `str` | `None` | System-level guidance such as persona style or business logic rules. |
| `store` | `bool` | `True` | Whether to persist the response session on the server for context chaining. |
| `previous_response_id` | `str` | `None` | Response ID prefixed with `resp_` to continue an existing conversation thread. |
| `text.format` | `dict` | `None` | Output schema enforcement (e.g., `{"type": "json_object"}`). |
| `reasoning` | `dict` | `{"effort": "medium"}` | Reasoning depth control; options are `"low"`, `"medium"`, or `"high"`. |
| `max_output_tokens` | `int` | `None` | Hard cap on total tokens generated, including internal reasoning tokens. |

**Response Object Fields**

| Field | Type | Description |
|:---|:---|:---|
| `id` | `str` | Unique server-side identifier for the response, prefixed with `resp_`. |
| `object` | `str` | API object type; always `"response"`. |
| `model` | `str` | Identifier of the model that produced the output. |
| `output` | `list[dict]` | Array of typed output items (reasoning blocks and message blocks). |
| `output_text` | `str` | Convenience accessor returning the final generated text content. |

```python
from openai import OpenAI

# 1. Initialize the OpenAI client
client = OpenAI()

# 2. Execute a stateful generation request with JSON output enforcement
response = client.responses.create(
    model="gpt-5.5",
    reasoning={"effort": "low"},
    instructions="Format outputs as clean JSON data structures.",
    input="Classify this sequence: 'System-level graph compiled successfully.'",
    text={"format": {"type": "json_object"}}
)

# 3. Access the final text directly via the output helper
print(response.output_text)

# 4. Chain a follow-up turn by referencing the previous response ID
followup = client.responses.create(
    model="gpt-5.5",
    previous_response_id=response.id,
    input="Now summarize your classification in one sentence."
)
print(followup.output_text)
```

---

### `def client.conversations.create(...)`

Creates a durable, long-running conversation thread that persists state on the server, enabling multi-turn dialogue without manually resending history.

```python
from openai import OpenAI

# 1. Initialize client
client = OpenAI()

# 2. Create a persistent conversation thread
conversation = client.conversations.create(model="gpt-5.5")
print(f"Conversation ID: {conversation.id}")

# 3. Send a message within the conversation context
response = client.responses.create(
    model="gpt-5.5",
    input="What is multi-head attention?",
    previous_response_id=conversation.id
)
print(response.output_text)
```

</details>

---

<details>
<summary><strong>Realtime Streaming API</strong></summary>

&nbsp;

> Stateful WebSocket/WebRTC connection for sub-200ms text and audio generation with dynamic session configuration.

---

### `async def client.realtime.connect(model)`

Establishes a persistent, stateful WebSocket connection to the Realtime API engine for low-latency audio and text streaming.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | Realtime model identifier (e.g., `"gpt-realtime-2"`). |

#### `await connection.session.update(session)`

Dynamically updates the active session configuration, including output modalities and voice settings.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `session.output_modalities` | `list[str]` | *Required* | Output types to generate; options are `"text"` and `"audio"`. |
| `session.voice` | `str` | `None` | Voice model for audio synthesis (e.g., `"shimmer"`). |

#### `await connection.conversation.item.create(item)`

Sends a conversation message item over the active WebSocket connection.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `item.type` | `str` | *Required* | Item type; use `"message"` for user input turns. |
| `item.role` | `str` | *Required* | Message role: `"user"` or `"assistant"`. |
| `item.content` | `list[dict]` | *Required* | Content blocks; use `{"type": "input_text", "text": "..."}` for text input. |

```python
import asyncio
from openai import AsyncOpenAI

async def stream_audio_session():
    # 1. Initialize async client
    client = AsyncOpenAI()

    # 2. Open stateful WebSocket connection
    async with client.realtime.connect(model="gpt-realtime-2") as connection:

        # 3. Configure session for text + audio output with a specific voice
        await connection.session.update(
            session={
                "output_modalities": ["text", "audio"],
                "voice": "shimmer"
            }
        )

        # 4. Send a user message item to the active session
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Synthesize a welcome message."}],
            }
        )

        # 5. Trigger model response generation
        await connection.response.create()

        # 6. Stream and print real-time text deltas as they arrive
        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

asyncio.run(stream_audio_session())
```

</details>

---

<details>
<summary><strong>Large-Scale Ingestion & Fine-Tuning</strong></summary>

&nbsp;

> APIs for uploading datasets, running high-throughput asynchronous batches, and initiating supervised or preference-based fine-tuning jobs.

---

### `def client.files.create(file, purpose, expires_after)`

Registers a file on the server for retrieval-augmented generation, batch processing, or model fine-tuning.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `file` | `BinaryIO` | *Required* | File object opened in binary mode for upload. |
| `purpose` | `str` | *Required* | Intended use: `"assistants"`, `"batch"`, `"fine-tune"`, or `"user_data"`. |
| `expires_after` | `dict` | `None` | Optional expiry policy (e.g., `{"days": 7}`) for automatic deletion. |

---

### `def client.uploads.create(bytes, filename, mime_type, purpose)`

Initializes a chunked multipart upload session for large assets up to 8 GB.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `bytes` | `int` | *Required* | Total byte size of the file being uploaded. |
| `filename` | `str` | *Required* | Destination filename used to identify the upload on the server. |
| `mime_type` | `str` | *Required* | MIME type of the file (e.g., `"application/jsonl"`). |
| `purpose` | `str` | *Required* | Intended use: `"assistants"`, `"vision"`, `"batch"`, or `"fine-tune"`. |

---

### `def client.batches.create(input_file_id, endpoint, completion_window)`

Submits a high-volume batch of requests for asynchronous processing at reduced cost.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `input_file_id` | `str` | *Required* | File ID of the `.jsonl` batch input file uploaded via `client.files.create`. |
| `endpoint` | `str` | *Required* | API endpoint for batch requests (e.g., `"/v1/responses"`). |
| `completion_window` | `str` | `"24h"` | Maximum processing window; all requests complete within this duration. |

---

### `def fine_tuning.jobs.create(model, training_file, hyperparameters, method)`

Initiates a fine-tuning job on a pre-trained model using supervised, preference, or reinforcement learning objectives.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `model` | `str` | *Required* | Base model to fine-tune (e.g., `"gpt-4o-mini"`). |
| `training_file` | `str` | *Required* | File ID of the uploaded training dataset in `.jsonl` format. |
| `hyperparameters` | `dict` | `None` | Training settings such as `n_epochs`, `batch_size`, and `learning_rate_multiplier`. |
| `method` | `str` | `"supervised"` | Training objective: `"supervised"` (SFT), `"dpo"`, or `"reinforcement"`. |

```python
from openai import OpenAI

client = OpenAI()

# 1. Upload the training dataset
with open("training_data.jsonl", "rb") as f:
    training_file = client.files.create(file=f, purpose="fine-tune")

print(f"Training file ID: {training_file.id}")

# 2. Submit a supervised fine-tuning job
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini",
    training_file=training_file.id,
    method="supervised",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 8,
        "learning_rate_multiplier": 0.1
    }
)

print(f"Fine-tuning job ID: {job.id}, Status: {job.status}")

# 3. Poll job status until complete
import time
while job.status not in ("succeeded", "failed"):
    time.sleep(30)
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job.status}")

print(f"Fine-tuned model: {job.fine_tuned_model}")
```

</details>

---

<details>
<summary><strong>Video Synthesis API</strong></summary>

&nbsp;

> Programmatic video generation, expansion, and editing powered by the Sora model family.

> ⚠️ **Deprecation Notice:** The Sora 2 model family and Videos API are scheduled for retirement on **September 24, 2026**.

---

### `def client.videos.create(prompt, model, size, seconds)`

Submits a programmatic video generation task from a text prompt, returning a job object with status tracking.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `prompt` | `str` | *Required* | Text description of the video content, style, and cinematography. |
| `model` | `str` | *Required* | Sora model variant: `"sora-2"` for standard, `"sora-2-pro"` for up to 1080p. |
| `size` | `str` | *Required* | Output resolution (e.g., `"1280x720"`, `"1920x1080"`). |
| `seconds` | `str` | *Required* | Generation duration: `"4"`, `"8"`, or `"12"` seconds. |

**Duration & Extension Constraints**

| Constraint | Value |
|:---|:---|
| Base generation durations | `4s`, `8s`, `12s` |
| Maximum extension per segment | `20s` via `client.videos.extend` |
| Maximum total video length | `120s` |
| `sora-2-pro` max resolution | `1080p` landscape / portrait |

```python
from openai import OpenAI
import time

# 1. Initialize the OpenAI client
client = OpenAI()

# 2. Submit a video generation job
video_job = client.videos.create(
    prompt="A wide pan of a highly modular robotic assembly line, cinematic lighting, 4k",
    model="sora-2-pro",
    size="1280x720",
    seconds="8"
)

print(f"Enqueued Video Job: {video_job.id}, Status: {video_job.status}")

# 3. Poll until generation completes
while video_job.status not in ("succeeded", "failed"):
    time.sleep(10)
    video_job = client.videos.retrieve(video_job.id)
    print(f"Status: {video_job.status}")

# 4. Extend the video by an additional 8 seconds
if video_job.status == "succeeded":
    extended = client.videos.extend(
        video_id=video_job.id,
        seconds="8"
    )
    print(f"Extended job ID: {extended.id}")
```

</details>

---

<details>
<summary><strong>Agentic Orchestration</strong></summary>

&nbsp;

> Multi-agent coordination framework with tool calling, handoffs, sandboxed execution, and global SDK configuration.

---

### `class Agent(name, instructions, tools)`

Configures a named AI agent with a behavioral persona and a set of callable tools for task execution.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `name` | `str` | *Required* | Human-readable identifier for the agent instance. |
| `instructions` | `str` | *Required* | System-level behavioral guidance defining the agent's role and constraints. |
| `tools` | `list` | `[]` | List of `@function_tool`-decorated callables or sub-agent tools available to the agent. |

#### `.as_tool()`

Wraps the agent as a callable tool for use within a manager/coordinator agent (Manager Pattern).

---

### `async def Runner.run(agent, input)`

Executes a single-agent or multi-agent workflow asynchronously, returning a result object containing the final output.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `agent` | `Agent` | *Required* | The entry-point agent that receives the initial input. |
| `input` | `str` | *Required* | Natural-language task description or query sent to the agent. |

**Returns:** `RunResult` with `.final_output` containing the agent's last response.

---

### `@function_tool`

Decorator that registers a Python function as a structured, schema-validated tool callable by an agent.

```python
from pydantic import BaseModel
from agents import Agent, Runner, function_tool
import asyncio

# 1. Define a structured output schema using Pydantic
class SystemStatus(BaseModel):
    node_id: str
    is_active: bool
    utilization: float

# 2. Register a Python function as an agent-callable tool
@function_tool
def check_node_telemetry(node_id: str) -> SystemStatus:
    # Query system telemetry (replace with real data source)
    return SystemStatus(node_id=node_id, is_active=True, utilization=0.78)

# 3. Configure a specialized agent with the tool attached
telemetry_agent = Agent(
    name="Telemetry Specialist",
    instructions="Analyze system metrics and check node utilization levels.",
    tools=[check_node_telemetry]
)

# 4. Run the agentic workflow and print the result
async def run_telemetry_check():
    result = await Runner.run(
        agent=telemetry_agent,
        input="Check status for node_abc123"
    )
    print(result.final_output)

asyncio.run(run_telemetry_check())
```

---

### Global SDK Configuration Methods

Configure default behaviors, API targets, and tracing for all agent runs.

| Method | Signature | Description |
|:---|:---|:---|
| `set_default_openai_client` | `set_default_openai_client(client)` | Sets a pre-configured `AsyncOpenAI` instance as the global default for all agent requests. |
| `set_default_openai_api` | `set_default_openai_api(api_shape)` | Switches the generation backend between the Responses API and legacy Chat Completions API. |
| `set_tracing_disabled` | `set_tracing_disabled(bool)` | Globally enables or disables transaction tracing for agent step tracking. |

---

### `class RunConfig(trace_include_sensitive_data)`

Controls per-run configuration, including whether sensitive inputs and outputs are included in exported trace logs.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `trace_include_sensitive_data` | `bool` | `True` | If `False`, strips sensitive input/output data from exported traces. |

---

### Sandboxed Compute — `SandboxAgent`

### `class SandboxAgent(name, instructions)`

Configures an agent that executes commands, manages dependencies, and edits files within an isolated compute container.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `name` | `str` | *Required* | Identifier for the sandboxed agent instance. |
| `instructions` | `str` | *Required* | Behavioral guidance defining permitted sandbox actions. |

---

### `class Manifest(mounts)`

Defines a workspace layout for a sandboxed agent, including cloud storage mounts and local directory bindings.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `mounts` | `list[dict]` | `[]` | List of `{"source": "...", "target": "..."}` mount specifications. |

```python
from agents import Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent
from agents.sandbox.sandboxes import UnixLocalSandboxClient

# 1. Configure a sandboxed agent for safe file system inspection
sandbox_agent = SandboxAgent(
    name="Code Analyst",
    instructions="Inspect file systems and execute safe sandbox commands."
)

# 2. Define a workspace manifest with S3 log mount
workspace_manifest = Manifest(
    mounts=[{
        "source": "s3://production-telemetry-bucket/logs",
        "target": "/workspace/logs"
    }]
)

# 3. Run the agent with sensitive data excluded from traces
import asyncio

async def run_sandbox():
    result = await Runner.run(
        agent=sandbox_agent,
        input="List all log files modified in the last 24 hours.",
        run_config=RunConfig(trace_include_sensitive_data=False)
    )
    print(result.final_output)

asyncio.run(run_sandbox())
```

**Supported Sandbox Providers**

| Provider | Use Case |
|:---|:---|
| `Daytona` | Full cloud development environments |
| `E2B` | Lightweight ephemeral code sandboxes |
| `Modal` | GPU-accelerated compute workloads |
| `Vercel` | Edge-native serverless execution |
| `UnixLocalSandboxClient` | Local development and testing |

</details>

---

## Part III — Hybrid Pipeline Architecture

<details>
<summary><strong>End-to-End Hybrid Pipeline</strong></summary>

&nbsp;

> Combines on-premise TensorFlow processing with cloud API reasoning and agentic orchestration.

```
+-------------------------------------------------------------+
|              Local TensorFlow / Keras Pipeline              |
|                                                             |
|   Raw Sensors ---> [ tf.keras.layers.MultiHeadAttention ]   |
|                                |                            |
|                                v                            |
|                    [ tf.keras.Model.predict() ]             |
+-------------------------------------------------------------+
                                 |
                                 | (Local Feature Vector)
                                 v
+-------------------------------------------------------------+
|                API-Driven Orchestration Layer               |
|                                                             |
|   Vector Data ---> [ client.files.create(purpose="RAG") ]   |
|                                |                            |
|                                v                            |
|                    [ client.responses.create() ]            |
|                                |                            |
|                                v                            |
|                    [ openai-agents Execution ]              |
+-------------------------------------------------------------+
```

**Stage Breakdown**

| Stage | Tool | Purpose |
|:---|:---|:---|
| 1. Local training | `tf.GradientTape` + custom Keras layers | Train proprietary vision/sequence models on-premise |
| 2. Feature extraction | `tf.keras.Model.predict()` | Generate embeddings or classifications from raw inputs |
| 3. Data ingestion | `client.files.create(purpose="user_data")` | Upload processed feature vectors to cloud storage |
| 4. Cloud reasoning | `client.responses.create()` | Run contextual reasoning over uploaded structured data |
| 5. Agentic execution | `Runner.run(agent, input)` | Orchestrate multi-step tasks, tool calls, and sandbox commands |

```python
import os
import tensorflow as tf
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio
import json

# ── Stage 1: Local feature extraction ────────────────────────────────────────

# 1. Load and run a local TensorFlow classification model
local_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(512,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

sensor_input = tf.random.normal((1, 512))
feature_vector = local_model.predict(sensor_input)
print(f"Local feature vector shape: {feature_vector.shape}")

# ── Stage 2: Upload features to cloud ────────────────────────────────────────

client = OpenAI()

# 2. Serialize feature vector and upload with "user_data" purpose
feature_payload = json.dumps({"features": feature_vector.tolist()})
feature_bytes = feature_payload.encode("utf-8")

import io
feature_file = client.files.create(
    file=io.BytesIO(feature_bytes),
    purpose="user_data"
)
print(f"Uploaded file ID: {feature_file.id}")

# ── Stage 3: Cloud reasoning via Responses API ────────────────────────────────

# 3. Send features to cloud model for contextual interpretation
response = client.responses.create(
    model="gpt-5.5",
    instructions="You are a sensor anomaly analyst. Interpret the provided feature vector.",
    input=f"Analyze this sensor classification output: {feature_vector.tolist()}"
)
analysis = response.output_text
print(f"Cloud analysis: {analysis}")

# ── Stage 4: Agentic follow-up action ────────────────────────────────────────

@function_tool
def trigger_alert(severity: str, message: str) -> dict:
    # 4. Simulate downstream alerting system
    return {"alert_sent": True, "severity": severity, "message": message}

coordinator = Agent(
    name="Incident Coordinator",
    instructions="Based on sensor analysis, decide whether to trigger alerts.",
    tools=[trigger_alert]
)

async def run_pipeline():
    # 5. Pass cloud analysis into the agentic layer for final action
    result = await Runner.run(
        agent=coordinator,
        input=f"Cloud analysis result: {analysis}. Determine alert severity."
    )
    print(f"Agent output: {result.final_output}")

asyncio.run(run_pipeline())
```

</details>

---

*Generated by Claude · Anthropic · Reference covers TensorFlow/Keras SDK and OpenAI API-Driven SDK paradigms for hybrid generative AI engineering.*