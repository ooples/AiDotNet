---
title: "PositionalEncodingType"
description: "Represents the type of positional encoding used in transformer attention layers."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents the type of positional encoding used in transformer attention layers.

## For Beginners

Transformers process all tokens in parallel, so they need
a way to know which position each token is at (first word, second word, etc.).

Different approaches:

- **Sinusoidal:** The original approach from "Attention Is All You Need" (2017)
- **Rotary (RoPE):** Used by Llama, Mistral, Phi, Gemma - encodes relative positions
- **ALiBi:** Used by BLOOM, MPT - adds a simple distance-based bias to attention scores
- **LearnedAbsolute:** Used by BERT, GPT-2 - learns position embeddings during training
- **None:** No positional encoding (for architectures that don't need it)

For modern LLMs, `Rotary` is the most common choice as of 2025-2026.

## How It Works

Positional encodings provide sequence position information to attention mechanisms,
which have no inherent notion of token order. Different encoding strategies trade off
between extrapolation to unseen lengths, computational cost, and compatibility with
KV-caching during inference.

## Fields

| Field | Summary |
|:-----|:--------|
| `ALiBi` | Attention with Linear Biases (ALiBi) from Press et al., 2022. |
| `LearnedAbsolute` | Learned absolute positional embeddings (BERT, GPT-2 style). |
| `None` | No positional encoding applied. |
| `Rotary` | Rotary Position Embedding (RoPE) from Su et al., 2021. |
| `Sinusoidal` | Sinusoidal positional encoding from the original Transformer paper (Vaswani et al., 2017). |

