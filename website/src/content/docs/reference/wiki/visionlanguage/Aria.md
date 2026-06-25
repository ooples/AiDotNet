---
title: "Aria<T>"
description: "Aria: mixture-of-experts multimodal model with efficient expert routing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Aria: mixture-of-experts multimodal model with efficient expert routing.

## For Beginners

Aria is a multimodal model that uses a "mixture of experts"
approach — instead of running every parameter for every token, it has 64 specialized expert
modules and routes each token to only the 8 most relevant ones. This makes the model very
efficient (only 3.9B of 25.3B parameters are active at once) while still being highly capable.
It can process images and long documents with up to 64K tokens of combined visual and text
context. Default values follow the original paper settings.

## How It Works

Aria (Rhymes AI, 2024) is a multimodal model that uses a Mixture-of-Experts (MoE) architecture
for efficient scaling. It has 25.3 billion total parameters but only activates 3.9 billion per
token, using 64 total experts with 8 active per token selected by a learned router. Each
visual and text token is routed to its top-scoring experts, allowing different experts to
specialize in different aspects of visual understanding. Aria supports a 64K multimodal
context window for processing long sequences of interleaved images and text.

**References:**

- Paper: "Aria: An Open Multimodal Native Mixture-of-Experts Model" (Rhymes AI, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Aria's MoE-based multimodal architecture. |

