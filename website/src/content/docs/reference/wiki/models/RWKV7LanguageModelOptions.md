---
title: "RWKV7LanguageModelOptions<T>"
description: "Configuration options for the RWKV-7 \"Goose\" language model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the RWKV-7 "Goose" language model.

## For Beginners

RWKV-7 is a text generation model that processes text in linear time:

**Key Advantages:**

- Linear time complexity: O(n) vs O(n^2) for Transformers
- Constant memory per token during generation
- Competitive quality with Transformer models of similar size

**Architecture:**

1. Token embedding converts words to vectors
2. N RWKV-7 blocks process the sequence, each with:
- Time mixing: WKV-7 kernel with dynamic state evolution
- Channel mixing: SiLU-gated feed-forward network
3. RMS normalization for stability
4. LM head projects to vocabulary logits

**Key Innovation (WKV-7):**
Instead of fixed exponential decay (RWKV-6), the state transition matrices a_t and b_t
are learnable and data-dependent:
S_t = diag(sigmoid(a_t)) * S_{t-1} + sigmoid(b_t) * outer(k_t, v_t)

This allows the model to dynamically decide what to remember and forget.

**Typical Model Sizes:**

- 0.1B: modelDim=768, numLayers=12, numHeads=12
- 1.5B: modelDim=2048, numLayers=24, numHeads=32
- 7B: modelDim=4096, numLayers=32, numHeads=64

## How It Works

RWKV-7 is the seventh generation of the RWKV architecture, introducing expressive dynamic
state evolution that replaces the fixed exponential decay of previous versions with learnable,
data-dependent transition matrices.

**Reference:** Peng et al., "RWKV-7 Goose with Expressive Dynamic State Evolution", 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RWKV7LanguageModelOptions` | Initializes a new instance with default values. |
| `RWKV7LanguageModelOptions(RWKV7LanguageModelOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `FFNMultiplier` | Gets or sets the FFN expansion multiplier. |
| `MaxSequenceLength` | Gets or sets the maximum sequence length. |
| `ModelDimension` | Gets or sets the model dimension (d_model). |
| `NumHeads` | Gets or sets the number of heads per block. |
| `NumLayers` | Gets or sets the number of RWKV-7 blocks. |
| `VocabSize` | Gets or sets the vocabulary size. |

