---
title: "InstructBLIP<T>"
description: "InstructBLIP: instruction-tuned BLIP-2 for zero-shot generalization across vision-language tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

InstructBLIP: instruction-tuned BLIP-2 for zero-shot generalization across vision-language tasks.

## For Beginners

InstructBLIP adds instruction-following capability to the
BLIP-2 model by instruction-tuning the Q-Former to extract visual features that are
relevant to the given instruction. The instruction is fed to both the Q-Former (to guide
what visual information to extract) and the LLM (to guide text generation), enabling
zero-shot generalization across diverse vision-language tasks. Default values follow
the original paper settings.

## How It Works

InstructBLIP (Dai et al., NeurIPS 2023) instruction-tunes the Q-Former component of BLIP-2
to extract instruction-aware visual features. The instruction is fed to both the Q-Former
(to guide visual feature extraction) and the LLM (to guide text generation).

**References:**

- Paper: "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning" (Dai et al., NeurIPS 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using InstructBLIP's instruction-aware Q-Former architecture. |
| `GetExtraTrainableLayers` |  |

