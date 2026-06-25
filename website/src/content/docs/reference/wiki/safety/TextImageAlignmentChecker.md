---
title: "TextImageAlignmentChecker<T>"
description: "Checks semantic alignment between text descriptions and associated images to detect mismatched or deceptive text-image pairs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Multimodal`

Checks semantic alignment between text descriptions and associated images to detect
mismatched or deceptive text-image pairs.

## For Beginners

This module checks if what an image shows matches what the text
says about it. For example, if someone labels an image "cute puppy" but the image shows
something violent, this module catches that mismatch.

## How It Works

Analyzes whether the visual content of an image is consistent with its text description.
Uses feature extraction from both modalities to compute alignment scores. Misalignment
can indicate deceptive content (safe text paired with unsafe image), phishing, or
misinformation.

**References:**

- CLIP: Learning transferable visual models from natural language (OpenAI, 2021)
- OmniSafeBench-MM: Multimodal safety evaluation (2025)
- MM-SafetyBench: 13 scenarios for multimodal safety (ECCV 2024)
- Cross-modal jailbreak attacks on multimodal LLMs (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextImageAlignmentChecker(Double)` | Initializes a new text-image alignment checker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateAlignment(String,Tensor<>)` | Evaluates alignment between text and an associated image. |
| `EvaluateText(String)` |  |

