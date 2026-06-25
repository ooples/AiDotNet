---
title: "CogVLM2<T>"
description: "CogVLM2: improved visual expert architecture with GLM-4 backbone and video understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

CogVLM2: improved visual expert architecture with GLM-4 backbone and video understanding.

## For Beginners

CogVLM2 is the successor to CogVLM that upgrades the language
backbone from Vicuna to GLM-4 (or LLaMA-3) and adds video understanding. It keeps the
same "visual expert" approach — dedicated visual expert modules in every decoder layer with
separate attention weights for image tokens versus text tokens. The key additions are temporal
attention for processing video frame sequences and improved training that gives it stronger
performance on both image and video tasks. Default values follow the original paper
settings.

## How It Works

CogVLM2 (Hong et al., 2024) improves upon CogVLM with GLM-4 (or LLaMA-3) as the language
backbone, enhanced visual expert modules, and video understanding capabilities. It maintains
the deep fusion approach of visual experts in every decoder layer while adding temporal
attention for video frame sequences.

**References:**

- Paper: "CogVLM2: Visual Language Models for Image and Video Understanding" (Hong et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using CogVLM2's improved visual expert + video architecture. |

