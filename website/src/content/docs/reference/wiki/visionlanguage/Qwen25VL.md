---
title: "Qwen25VL<T>"
description: "Qwen2.5-VL: visual agent with 1hr+ video, bounding box/point localization capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Qwen2.5-VL: visual agent with 1hr+ video, bounding box/point localization capabilities.

## For Beginners

Qwen2.5-VL extends Qwen2-VL with visual agent capabilities —
it can not only understand images and videos but also interact with visual interfaces
like a human user would. It supports understanding videos over 1 hour long, locating
objects with bounding boxes and point coordinates, and excels at reading complex documents
and charts. The model works as a visual agent that can navigate UIs, understand screenshots,
and perform actions based on what it sees. Available in multiple sizes for different
compute budgets. Default values follow the original paper settings.

## How It Works

Qwen2.5-VL extends Qwen2-VL with enhanced visual agent capabilities including support for
1hr+ video understanding, bounding box and point localization, and stronger document/chart
comprehension. Uses M-RoPE positional encoding and Qwen2.5 language backbone.

**References:**

- Paper: "Qwen2.5-VL Technical Report" (2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Qwen2.5-VL's visual-agent architecture. |
| `GetExtraTrainableLayers` |  |

