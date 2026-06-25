---
title: "Moondream<T>"
description: "Moondream: tiny yet capable VLM for resource-constrained environments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Moondream: tiny yet capable VLM for resource-constrained environments.

## For Beginners

Moondream is one of the smallest usable vision-language models
available — with under 2 billion parameters, it can run on devices with very limited
memory and compute. Despite its tiny size, it can describe images, answer questions about
photos, and perform basic visual reasoning. It uses a SigLIP vision encoder paired with
a compact Phi-based language model. Think of it as the model you use when you need
on-device visual AI but have very limited resources — like a Raspberry Pi or an
embedded system. Default values follow the original paper settings.

## How It Works

Moondream (Korrapati, 2024) is an extremely compact vision-language model designed for
resource-constrained environments. Despite having fewer than 2 billion parameters, it
achieves surprisingly capable visual understanding by using an efficient SigLIP vision
encoder with a compact Phi-based language model. Its small footprint makes it suitable
for edge deployment, embedded systems, and scenarios where larger models would be
impractical due to memory or compute limitations.

**References:**

- Paper: "Moondream: A Tiny Vision Language Model" (Vikhyat Korrapati, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Moondream's lightweight SigLIP + Phi-1.5 architecture. |

