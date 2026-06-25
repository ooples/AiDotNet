---
title: "Monkey<T>"
description: "Monkey: high-resolution VLM with multi-level description generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Monkey: high-resolution VLM with multi-level description generation.

## For Beginners

Monkey focuses on two things that turn out to be critical for
vision-language models: high image resolution and high-quality text labels. Most models
downscale images to a fixed small size, losing fine details. Monkey instead processes
images at up to 1344x896 resolution by splitting them into patches. It also uses
carefully curated training data with detailed, accurate text descriptions. The result
is a model that can generate descriptions at multiple levels of detail — from a brief
one-line caption to a rich multi-paragraph description — and excels at understanding
fine visual details like small text and intricate patterns. Default values follow the
original paper settings.

## How It Works

Monkey (Li et al., 2024) demonstrates that image resolution and text label quality are crucial
for multimodal model performance. It processes images at high resolution (up to 1344x896) by
dividing them into patches and encoding each at full resolution, then generates multi-level
descriptions — from brief captions to detailed paragraph-level descriptions — using a
carefully curated training dataset with high-quality text annotations.

**References:**

- Paper: "Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Monkey's multi-level high-resolution description architecture. |

