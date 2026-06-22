# Keras GenAI SDK

*A scannable reference for the Keras 3 multi-backend SDK and the KerasHub generative AI extension library, covering backend configuration, task/backbone/preprocessor architecture, generation, sampling, tokenization, diffusion, and low-level ops.*

---

<details>
<summary><strong>1. Backend Configuration & Execution Model</strong></summary>

### `KERAS_BACKEND`
Environment variable (or `~/.keras/keras.json` entry) that selects the mathematical engine used to build and execute graphs.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `KERAS_BACKEND` | `str` | *Required* | One of `"jax"`, `"tensorflow"`, or `"torch"`. Determines graph construction, compilation, and autodiff pathway. |

```python
# 1. Set the active backend before importing keras
import os
os.environ["KERAS_BACKEND"] = "jax"

# 2. Import keras after the backend is configured
import keras
```

---

### `TF_USE_LEGACY_KERAS`
Environment variable that redirects `from tensorflow import keras` to the legacy Keras 2 engine (via the standalone `tf_keras` package) instead of Keras 3.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `TF_USE_LEGACY_KERAS` | `int` | `0` | Set to `1` to force legacy Keras 2 graph compilation and state management. |

```python
# 1. Install the legacy package
# pip install tf_keras

# 2. Force legacy execution
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 3. Import as usual — now resolves to Keras 2
from tensorflow import keras
```

**Backend Compatibility Matrix**

| Backend Runtime Engine | Target Keras Core Version | Min. Package Version | Primary Execution Modes |
|:---|:---|:---:|:---|
| JAX | Keras 3.0+ | `jax==0.4.20` | Stateless functional trace execution, XLA graph compilation |
| TensorFlow | Keras 3.0+ | `tensorflow>=2.16.1` | Autograph conversion, stateful variable tracking, `tf.data` integration |
| PyTorch | Keras 3.0+ | `torch~=2.1.0` | Eager execution dispatch, imperative autodiff, native `torch.compile` |
| TensorFlow (Legacy) | Keras 2.13–2.15 | `tensorflow~=2.13.0` | Fused graph operators, standalone state management |

**Benchmark — Keras 3 vs. Legacy Keras 2**

| Model Family | Throughput Increase | Hardware |
|:---|:---:|:---|
| BERT Training | `> 100%` | TPU / GPU accelerated instances |
| Stable Diffusion Training | `> 150%` | V100 / A100 GPU clusters |
| SegmentAnything Inference | `380%` | Consolidated inference runtimes |

</details>

---

<details>
<summary><strong>2. KerasHub Architectural Hierarchy</strong></summary>

KerasHub enforces a strict separation of concerns across four layers:

| Layer | Base Class | Responsibility |
|:---|:---|:---|
| **Task** | `keras.Model` | End-to-end entry point; combines preprocessing + modeling. Accepts raw inputs, returns predictions. |
| **Backbone** | `keras_hub.models.Backbone` | Task-agnostic architecture mapping preprocessed tensors to latent representations. |
| **Preprocessor** | `keras_hub.models.Preprocessor` | Packages tokenizers, normalizers, and formatters into a reusable pipeline. |
| **Tokenizer / Converter** | — | Lowest-level layer mapping raw modalities (text/image) to numerical tensors. |

This separation allows CPU-bound preprocessing (`tf.data`) to run out-of-band while the backbone executes on accelerators.

**Generative Model Class Registry**

| Task Class | Backbone | Modality | Construction |
|:---|:---|:---|:---|
| `GPT2CausalLM` | `GPT2Backbone` | Autoregressive causal text | `backbone, preprocessor=None` |
| `QwenCausalLM` | `QwenBackbone` | Autoregressive causal text | `backbone, preprocessor=None` |
| `LlamaCausalLM` | `LlamaBackbone` | Autoregressive causal text | `backbone, preprocessor=None` |
| `Phi3CausalLM` | `Phi3Backbone` | Autoregressive causal text | `backbone, preprocessor=None` |
| `OPTCausalLM` | `OPTBackbone` | Autoregressive causal text | `backbone, preprocessor=None` |
| `QwenMoeCausalLM` | `QwenMoeBackbone` | Autoregressive Mixture-of-Experts text | `backbone, preprocessor=None` |
| `Gemma3CausalLM` | `Gemma3Backbone` | Multimodal vision & text | `preprocessor, backbone` |
| `T5GemmaSeq2SeqLM` | `T5GemmaBackbone` | Sequence-to-sequence conditional generation | `backbone, preprocessor=None` |
| `MoonshineAudioToText` | `MoonshineBackbone` | Audio transcription / translation | `backbone, preprocessor=None` |

</details>

---

<details>
<summary><strong>3. Model Loading & Initialization</strong></summary>

### `def from_preset(preset, load_weights=True, dtype=None)`
Factory method exposed on all Task, Backbone, and Preprocessor classes. Resolves a model identifier, downloads config/weights, and returns an initialized instance.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `preset` | `str` | *Required* | A built-in preset name (`"bert_base_en"`), local path (`"./gpt2_base_en"`), Kaggle handle (`"kaggle://user/bert/keras/bert_base_en"`), or Hugging Face handle (`"hf://user/bert_base_en"`). |
| `load_weights` | `bool` | `True` | If `False`, builds the architecture with randomly initialized weights — useful for pretraining from scratch. |
| `dtype` | `str` | `None` | Compute dtype override, e.g. `"bfloat16"` or `"float16"`. |

```python
# 1. Load a pretrained causal LM task from a hub preset
causal_model = keras_hub.models.CausalLM.from_preset(
    "gemma2_2b_en",
    load_weights=True,
    dtype="bfloat16"
)

# 2. Generate text immediately — preprocessing is handled internally
output = causal_model.generate("Keras is an API designed for")
```

---

### `class keras_hub.models.QwenMoeBackbone(...)`
Directly instantiates a custom Mixture-of-Experts backbone, bypassing presets entirely.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `vocabulary_size` | `int` | *Required* | Size of the token vocabulary. |
| `num_layers` | `int` | *Required* | Number of transformer blocks. |
| `num_query_heads` | `int` | *Required* | Number of attention query heads. |
| `num_key_value_heads` | `int` | *Required* | Number of key/value heads (for GQA). |
| `hidden_dim` | `int` | *Required* | Model hidden dimensionality. |
| `intermediate_dim` | `int` | *Required* | Dense feed-forward intermediate size. |
| `moe_intermediate_dim` | `int` | *Required* | Per-expert feed-forward intermediate size. |
| `shared_expert_intermediate_dim` | `int` | *Required* | Intermediate size of the shared expert. |
| `num_experts` | `int` | *Required* | Total number of experts in each MoE layer. |
| `top_k` | `int` | *Required* | Number of experts routed to per token. |
| `max_sequence_length` | `int` | *Required* | Maximum supported input sequence length. |

```python
# 1. Construct a custom MoE backbone from scratch
moe_backbone = keras_hub.models.QwenMoeBackbone(
    vocabulary_size=151936,
    num_layers=28,
    num_query_heads=16,
    num_key_value_heads=8,
    hidden_dim=2048,
    intermediate_dim=4096,
    moe_intermediate_dim=128,
    shared_expert_intermediate_dim=4096,
    num_experts=60,
    top_k=4,
    max_sequence_length=4096
)
```

</details>

---

<details>
<summary><strong>4. Generation & Scoring</strong></summary>

### `def generate(inputs, max_length=None, stop_token_ids="auto", strip_prompt=False)`
Orchestrates tokenization, next-token prediction, and autoregressive decoding for a Task model.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `inputs` | `str \| Tensor \| tf.data.Dataset` | *Required* | Raw strings, token tensors, or a dataset pipeline. Preprocessed internally if a preprocessor is attached. |
| `max_length` | `int` | `None` | Max total sequence length (prompt + completion). Defaults to the preprocessor's configured length. |
| `stop_token_ids` | `"auto" \| list[int]` | `"auto"` | `"auto"` stops at the tokenizer's `end_token_id`; otherwise pass explicit token IDs. |
| `strip_prompt` | `bool` | `False` | If `True`, returns only newly generated text, excluding the input prompt. |

```python
# 1. Run text generation with custom decoding controls
completed_text = causal_model.generate(
    inputs="Keras is an API designed for",
    max_length=64,
    stop_token_ids="auto",
    strip_prompt=True
)

# 2. Print the result
print(completed_text)
```

---

### `def score(token_ids, padding_mask=None, scoring_mode="loss", target_ids=None, layer_intercept_fn=None)`
Evaluates exact token transition probabilities or loss for a sequence (available on Gemma-family models).

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `token_ids` | `Tensor[batch, num_tokens]` | *Required* | Complete token sequence to evaluate. |
| `padding_mask` | `Tensor` | `None` | Binary mask of tokens to preserve; defaults to all-ones via `keras.ops.ones()`. |
| `scoring_mode` | `"loss" \| "logits"` | `"loss"` | Returns categorical cross-entropy loss or raw vocabulary logits. |
| `target_ids` | `Tensor` | `None` | Optional target labels for computing loss over specific spans. |
| `layer_intercept_fn` | `Callable` | `None` | Interpretability hook intercepting activations; `-1` = embedding layer, `≥0` = transformer block index. |

```python
# 1. Evaluate sequence loss for a batch of token sequences
token_losses = causal_model.score(
    token_ids=batched_tokens,
    padding_mask=keras.ops.ones(shape=(2, 64)),
    scoring_mode="loss"
)

# 2. Inspect per-token loss values
print(token_losses)
```

</details>

---

<details>
<summary><strong>5. Samplers API</strong></summary>

Each architecture registers a default sampler: **`"greedy"`** (Gemma, Gemma3, Qwen, QwenMoe, T5-Gemma) or **`"top_k"`** (LLaMA, Phi-3, BART). Override via `.compile(sampler=...)`.

### `class keras_hub.samplers.GreedySampler()`
Selects the highest-probability token at each decoding step: `t = argmax_i P(w_i)`.

No configurable parameters.

---

### `class keras_hub.samplers.BeamSampler(num_beams, return_all_beams=False)`
Tracks the `num_beams` most probable cumulative-probability paths, pruning low-scoring sequences.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `num_beams` | `int` | *Required* | Number of candidate beams to retain at each step. |
| `return_all_beams` | `bool` | `False` | If `True`, returns all final beams instead of just the top one. |

---

### `class keras_hub.samplers.RandomSampler(seed=None, temperature=1.0)`
Samples randomly from the full vocabulary based on raw softmax probabilities.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `seed` | `int` | `None` | Random seed for reproducibility. |
| `temperature` | `float` | `1.0` | Scales distribution sharpness; higher values increase randomness. |

---

### `class keras_hub.samplers.TopKSampler(k, seed=None, temperature=1.0)`
Filters to the `k` most likely tokens, renormalizes, and samples from that restricted pool.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `k` | `int` | *Required* | Number of top tokens to retain in the sampling pool. |
| `seed` | `int` | `None` | Random seed for reproducibility. |
| `temperature` | `float` | `1.0` | Scales distribution sharpness. |

---

### `class keras_hub.samplers.TopPSampler(p, k=None, seed=None)`
Selects the smallest token set whose cumulative probability exceeds threshold `p`, dynamically scaling pool size.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `p` | `float` | *Required* | Cumulative probability threshold. |
| `k` | `int` | `None` | Optional hard cap on pool size before applying `p`. |
| `seed` | `int` | `None` | Random seed for reproducibility. |

---

### `class keras_hub.samplers.ContrastiveSampler(k, alpha)`
Balances prediction probability against a repetition penalty: `score(v) = (1-α)·P(v) - α·max_j Sim(v, s_j)`.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `k` | `int` | *Required* | Candidate pool size considered at each step. |
| `alpha` | `float` | *Required* | Weight of the similarity penalty term, in `[0, 1]`. |

```python
# 1. Compile a task with a custom Top-P sampler
causal_model.compile(
    sampler=keras_hub.samplers.TopPSampler(p=0.9, k=50, seed=42)
)

# 2. Generation now uses the configured sampling strategy
output = causal_model.generate("Once upon a time")
```

</details>

---

<details>
<summary><strong>6. Tokenizers & Preprocessing</strong></summary>

**Pipeline flow:** Raw inputs → Preprocessing → Tokenizers → Sequence Packager → Model inputs (latent space).

### `class keras_hub.tokenizers.WordPieceTokenizer`
Implements the subword WordPiece algorithm used by BERT, with case folding, normalization, and OOV handling (`"[UNK]"`, `"[PAD]"` → index `0`).

### `class keras_hub.tokenizers.SentencePieceTokenizer`
Implements SentencePiece tokenization on raw, unsplit text; supports loading a custom `.proto` vocabulary file.

### `class keras_hub.tokenizers.BytePairTokenizer`
Implements Byte-Pair Encoding (BPE) by iteratively merging high-frequency byte pairs.

### `class keras_hub.tokenizers.ByteTokenizer`
Maps raw text directly to byte values, bypassing subword vocabulary tables.

### `class keras_hub.tokenizers.UnicodeCodepointTokenizer`
Maps text to integer Unicode codepoints for character-level modeling.

Utility functions `compute_word_piece_vocabulary()` and `compute_sentence_piece_proto()` build custom vocabulary files for training new tokenizers.

---

### `class keras_hub.layers.StartEndPacker(sequence_length, start_value, end_value, pad_value)`
Pads, truncates, and wraps tokenized sequences with boundary tokens for transformer input.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `sequence_length` | `int` | *Required* | Target fixed length for all output sequences. |
| `start_value` | `int` | *Required* | Token ID inserted at the start of the sequence (e.g. `[BOS]`). |
| `end_value` | `int` | *Required* | Token ID inserted at the end of the sequence (e.g. `[EOS]`). |
| `pad_value` | `int` | *Required* | Token ID used to pad sequences to `sequence_length`. |

```python
# 1. Build a sequence packager with BOS/EOS boundary tokens
sequence_packager = keras_hub.layers.StartEndPacker(
    sequence_length=128,
    start_value=tokenizer.token_to_id("[BOS]"),
    end_value=tokenizer.token_to_id("[EOS]"),
    pad_value=0
)

# 2. Apply the packager to tokenized inputs
packed_tensor = sequence_packager(tokenized_inputs)
```

---

### `class keras_hub.layers.ImageConverter`
Resizes, rescales, and offsets raw image tensors for vision model consumption. Supports channels-first/last and batched/unbatched (rank 3 or 4) inputs.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `image_size` | `tuple` | *Required* | Target `(height, width)` to resize images to. |
| `scale` | `float` | `1.0` | Multiplier applied to normalize pixel value ranges. |
| `offset` | `float` | `0.0` | Numerical offset to shift normalized pixel values (e.g. to `[-1, 1]`). |

```python
# 1. Build an ImageConverter from a model preset
image_preprocessor = keras_hub.layers.ImageConverter.from_preset(
    "resnet_50_imagenet",
    image_size=(224, 224),
    scale=1.0 / 127.5,
    offset=-1.0
)

# 2. Preprocess a batch of raw images
processed_images = image_preprocessor(raw_image_batch)
```

### `class keras_hub.layers.AudioConverter`
Normalizes and converts raw audio signals into spectral representations for audio model input.

</details>

---

<details>
<summary><strong>7. Generative Computer Vision — Stable Diffusion 3</strong></summary>

SD3 uses a Multimodal Diffusion Transformer (MMDiT) backbone to jointly process image and text features, exposed through three task classes.

| Task Class | Description |
|:---|:---|
| `StableDiffusion3TextToImage` | Generates images directly from text prompts. |
| `StableDiffusion3ImageToImage` | Modifies a reference image guided by a text prompt. |
| `StableDiffusion3Inpaint` | Inpaints/edits a masked image region guided by a text prompt. |

```python
# 1. Load a text-to-image pipeline at reduced precision
sd3_pipeline = keras_hub.models.StableDiffusion3TextToImage.from_preset(
    "stable_diffusion_3_medium",
    dtype="float16"
)

# 2. Generate an image from a prompt with guidance controls
generated_images = sd3_pipeline.generate(
    inputs={
        "prompts": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "negative_prompts": "green color"
    },
    num_steps=28,
    guidance_scale=7.0
)
```

### `class keras_hub.models.StableDiffusion3Backbone(...)`
Configuration parameters for the underlying MMDiT backbone.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `mmdit_patch_size` | `int` | `2` | Patch dimensions used to tokenize images for the transformer. |
| `mmdit_hidden_dim` | `int` | `1536` | Hidden dimensionality of the MMDiT layers. |
| `mmdit_num_layers` | `int` | `24` | Number of transformer blocks in the MMDiT backbone. |
| `mmdit_num_heads` | `int` | `24` | Number of attention heads in the MMDiT layers. |
| `mmdit_qk_norm` | `str \| None` | `"rms_norm"` / `None` | Query/key normalization; `"rms_norm"` for SD3.5, `None` for SD3.0. |
| `vae` | `keras.Model` | *Required* | Pretrained Variational Autoencoder mapping images to/from latent space. |
| `clip_l` | `keras.Model` | *Required* | Core CLIP-L text encoder. |
| `clip_g` | `keras.Model` | *Required* | Secondary CLIP-G text encoder for additional guidance. |
| `t5` | `keras.Model` | `None` | Optional T5-XXL encoder improving prompt alignment. |
| `latent_channels` | `int` | `16` | Number of channels in the latent representation. |
| `num_train_timesteps` | `int` | `1000` | Total noise steps used during training. |
| `shift` | `float` | `3.0` | Shift value for the timestep noise scheduler. |
| `image_shape` | `tuple` | `(1024, 1024, 3)` | Spatial dimensions and channels of the output image. |

**Generation-time parameters** (passed to `.generate()`):

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `num_steps` | `int` | `28` | Number of iterative denoising steps; higher improves quality at the cost of compute. |
| `guidance_scale` | `float` | `7.0` | Classifier-free guidance strength; higher follows the prompt more strictly. |
| `strength` | `float` | `1.0` | (img2img only) How much the input image is modified; `1.0` treats it as pure noise. |
| `negative_prompts` | `str` | `None` | Guides the model away from specific styles or elements. |

</details>

---

<details>
<summary><strong>8. Low-Level Ops — Custom Architectures (VAE / GAN)</strong></summary>

### `class keras.random.SeedGenerator(seed)`
Maintains a stateful, trace-safe seed container so random operations remain compatible with JAX/TF graph tracing while still producing varying values across calls.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `seed` | `int` | *Required* | Initial seed value for the stateful generator. |

```python
# 1. Define a custom VAE latent sampling layer
class LatentSampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector representing an image."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 2. Initialize a trace-safe seed generator
        self.seed_gen = keras.random.SeedGenerator(seed=1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = keras.ops.shape(z_mean)[0]
        latent_dim = keras.ops.shape(z_mean)[1]

        # 3. Draw epsilon from a standard normal distribution
        epsilon = keras.random.normal(
            shape=(batch_size, latent_dim),
            mean=0.0,
            stddev=1.0,
            seed=self.seed_gen
        )
        # 4. Apply the reparameterization trick: z = mean + std * epsilon
        return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon
```

**Trace-Safe Random Operations**

| Operation | Parameter Signature | Description |
|:---|:---|:---|
| `keras.random.normal()` | `shape, mean=0.0, stddev=1.0, dtype=None, seed=None` | Draws samples from a normal (Gaussian) distribution. |
| `keras.random.uniform()` | `shape, minval=0.0, maxval=1.0, dtype=None, seed=None` | Draws samples from a uniform distribution over `[minval, maxval)`. |
| `keras.random.beta()` | `shape, alpha, beta, dtype=None, seed=None` | Draws samples from a Beta distribution. |

---

### Straight-Through Estimator (VQ-VAE Quantization)
Uses `keras.ops.stop_gradient` to make the non-differentiable `argmin` codebook-selection step trainable via gradient bypass.

```python
# 1. Compute the quantized latent using the straight-through trick
continuous_latent = continuous_encoder_outputs
quantized_latent = continuous_latent + keras.ops.stop_gradient(
    discrete_codebook_vectors - continuous_latent
)
# 2. Forward pass outputs the discrete vectors (continuous terms cancel)
# 3. Backward pass routes gradients directly to continuous_encoder_outputs
```

---

### `def dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, flash_attention=None, attn_logits_soft_cap=None)`
Computes scaled dot-product attention, automatically supporting Multi-Head (MHA), Grouped-Query (GQA), or Multi-Query (MQA) attention based on input tensor shapes.

| Parameter | Type | Default | Description |
|:---|:---|:---:|:---|
| `query` | `Tensor` | *Required* | Query tensor. |
| `key` | `Tensor` | *Required* | Key tensor. |
| `value` | `Tensor` | *Required* | Value tensor. |
| `bias` | `Tensor` | `None` | Optional additive attention bias. |
| `mask` | `Tensor` | `None` | Optional attention mask. |
| `scale` | `float` | `None` | Optional scaling factor for attention logits. |
| `is_causal` | `bool` | `False` | If `True`, applies causal masking. |
| `flash_attention` | `bool` | `None` | If `True`, enables flash attention kernels where supported. |
| `attn_logits_soft_cap` | `float` | `None` | Optional soft cap applied to attention logits. |

```python
# 1. Compute causal flash attention
attention_output = keras.ops.dot_product_attention(
    query=query_tensor,
    key=key_tensor,
    value=value_tensor,
    is_causal=True,
    flash_attention=True
)
```

**Additional Standard Operations (`keras.ops`)**

| Operation | Core Arguments | Description |
|:---|:---|:---|
| `conv_transpose` | `inputs, kernel, strides, padding="same", data_format="channels_last"` | Transposed convolution (deconvolution) for upsampling latents to pixel images. |
| `batch_normalization` | `x, mean, variance, axis, offset=None, scale=None, epsilon=0.001` | Normalizes input tensors along a specified axis. |
| `binary_crossentropy` | `target, output, from_logits=False` | Binary cross-entropy loss; supports probabilities or raw logits. |
| `categorical_crossentropy` | `target, output, from_logits=False, axis=-1` | Categorical cross-entropy loss for multi-class classification. |

</details>