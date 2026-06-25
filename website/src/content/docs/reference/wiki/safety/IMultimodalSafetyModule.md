---
title: "IMultimodalSafetyModule<T>"
description: "Interface for multimodal safety modules that analyze cross-modal content interactions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Multimodal`

Interface for multimodal safety modules that analyze cross-modal content interactions.

## For Beginners

A multimodal safety module checks the combination of different
content types together. For example, someone might pair innocent-sounding text with a
harmful image to bypass safety checks — this module catches those cross-modal attacks.

## How It Works

Multimodal safety modules detect risks that arise from the interaction between different
content modalities — for example, safe text paired with an unsafe image, or cross-modal
attacks that exploit mismatches between text and image safety classifiers.

**References:**

- Cross-modal safety mechanism transfer failure in VLMs (2024, arxiv:2410.12662)
- OmniSafeBench-MM: 13 attacks, 15 defenses, 9 risk domains (2025, arxiv:2512.06589)
- MM-SafetyBench: 5,040 text-image pairs across 13 scenarios (ECCV 2024)
- Llama Guard 3 Vision: Multimodal safety classification (Meta, 2024, arxiv:2411.10414)

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateTextAudio(String,Vector<>,Int32)` | Evaluates a text-audio pair for cross-modal safety risks. |
| `EvaluateTextImage(String,Tensor<>)` | Evaluates a text-image pair for cross-modal safety risks. |

