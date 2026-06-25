---
title: "LLaVAOneVision15<T>"
description: "LLaVA-OneVision-1.5: fully open model outperforming Qwen2.5-VL on 18/27 benchmarks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

LLaVA-OneVision-1.5: fully open model outperforming Qwen2.5-VL on 18/27 benchmarks.

## For Beginners

LLaVA-OneVision 1.5 is the fully open successor that upgrades
the language backbone to Qwen2.5 and uses improved training data and strategies. It
supports single images, multi-image comparisons, and long video understanding with up
to 64 frames. Notably, it outperforms the much larger Qwen2.5-VL on 18 out of 27
benchmarks while being fully open-source with all training data, code, and model weights
publicly available. This makes it one of the most capable and accessible open multimodal
models. Default values follow the original paper settings.

## How It Works

LLaVA-OneVision-1.5 (Li et al., 2025) is the fully open successor to LLaVA-OneVision,
using Qwen2.5 as the language backbone with improved training data and strategy. It supports
single images, multi-image, and long video understanding with up to 64 frames.

**References:**

- Paper: "LLaVA-OneVision 1.5: Improved and Fully Open" (Li et al., 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using LLaVA-OneVision-1.5's improved unified architecture. |

