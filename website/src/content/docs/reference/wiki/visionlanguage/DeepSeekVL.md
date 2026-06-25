---
title: "DeepSeekVL<T>"
description: "DeepSeek-VL: real-world vision-language understanding with hybrid SigLIP + SAM-B encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

DeepSeek-VL: real-world vision-language understanding with hybrid SigLIP + SAM-B encoder.

## For Beginners

DeepSeek-VL uses two vision encoders working together instead
of just one: SigLIP captures the semantic meaning of the image (what objects are present),
while SAM-B captures fine-grained spatial details (exact shapes and boundaries). Their
outputs are merged through an MLP projector and fed into the DeepSeek language model for
text generation. This hybrid approach gives the model both high-level understanding and
pixel-level precision for real-world tasks like reading documents, understanding charts,
and answering detailed visual questions. Default values follow the original paper
settings.

## How It Works

DeepSeek-VL (Lu et al., 2024) uses a hybrid vision encoder combining SigLIP for semantic
understanding and SAM-B for fine-grained spatial features. The dual encoder outputs are merged
via MLP projection into the DeepSeek language model for real-world multimodal tasks.

**References:**

- Paper: "DeepSeek-VL: Towards Real-World Vision-Language Understanding" (Lu et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using DeepSeek-VL's hybrid vision encoder architecture. |

