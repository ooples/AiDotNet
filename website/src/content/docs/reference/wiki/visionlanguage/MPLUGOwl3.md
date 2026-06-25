---
title: "MPLUGOwl3<T>"
description: "mPLUG-Owl3: enhanced with hyper-attention for long visual sequences."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

mPLUG-Owl3: enhanced with hyper-attention for long visual sequences.

## For Beginners

mPLUG-Owl3 tackles one of the biggest challenges in vision-language
models: efficiently processing long sequences of images or video frames. Standard attention
has quadratic cost — processing twice as many visual tokens takes four times the compute.
mPLUG-Owl3 introduces "hyper-attention" that handles long visual sequences much more
efficiently, making it practical to process many images or long videos. It uses Qwen2 as
the language backbone and an enhanced visual abstractor that can handle extended sequences
without the usual performance bottleneck. Default values follow the original paper
settings.

## How It Works

mPLUG-Owl3 (Alibaba, 2024) introduces hyper-attention for efficiently processing long
visual sequences. It uses Qwen2 as the language backbone with an enhanced visual abstractor
that can handle extended visual token sequences without quadratic attention cost.

**References:**

- Paper: "mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using mPLUG-Owl3's hyper-attention architecture. |
| `GetExtraTrainableLayers` |  |

