---
title: "InternVL25<T>"
description: "InternVL2.5: improved training data and strategy over InternVL2 with InternLM2.5 backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

InternVL2.5: improved training data and strategy over InternVL2 with InternLM2.5 backbone.

## For Beginners

InternVL 2.5 refines InternVL2 with better training data and
improved training strategies while keeping the same pixel shuffle and dynamic resolution
architecture. The key improvements are in data curation — more carefully selected and
filtered training examples — and optimized training recipes that help the model learn
more effectively. It uses InternLM2.5 as the language backbone, achieving stronger
benchmark scores across diverse tasks including document understanding, chart reasoning,
and visual question answering. Default values follow the original paper settings.

## How It Works

InternVL2.5 (Chen et al., 2024) refines InternVL2 with improved training data curation and
strategy, using InternLM2.5 as the language backbone. It maintains the pixel shuffle and
dynamic resolution architecture while achieving stronger benchmark performance.

**References:**

- Paper: "Expanding Performance Boundaries of Open-Source Multimodal Models with InternVL 2.5" (Chen et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using InternVL2.5's improved pixel shuffle + dynamic resolution. |

