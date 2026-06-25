---
title: "Eagle25<T>"
description: "Eagle 2.5: extended context VLM with video understanding capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Eagle 2.5: extended context VLM with video understanding capabilities.

## For Beginners

Eagle 2.5 extends the original Eagle model with the ability
to understand videos and long sequences of images. Through specialized long-context
post-training, it can process many visual frames in sequence — useful for video
understanding, multi-page document analysis, and tasks that require tracking information
across many images. It maintains strong single-image performance while adding this
extended context capability. Default values follow the original paper settings.

## How It Works

Eagle 2.5 (NVIDIA, 2025) extends Eagle with long-context post-training to handle extended
visual sequences including video understanding. It uses specialized post-training techniques
to boost the model's ability to process long sequences of visual frames while maintaining
strong performance on single-image tasks. The model supports extended context windows for
processing multi-frame video content and long document sequences.

**References:**

- Paper: "Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models" (2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Eagle 2.5's adaptive tiling with long-context support. |

