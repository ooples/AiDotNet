---
title: "IInstructionTunedVLM<T>"
description: "Interface for instruction-tuned vision-language models that generate conversational responses from visual input conditioned on user instructions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for instruction-tuned vision-language models that generate conversational responses
from visual input conditioned on user instructions.

## How It Works

Instruction-tuned VLMs extend generative VLMs with instruction-following capabilities.
They are fine-tuned on visual instruction data to enable conversational AI about images.
Architectures include:

- MLP projection (LLaVA, InternVL): ViT → MLP connector → LLM
- Q-Former projection (MiniGPT-4): ViT → Q-Former → linear → LLM
- Cross-attention resampler (Qwen-VL): ViT → resampler → LLM
- Visual expert (CogVLM): ViT → visual expert modules in every LLM layer

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets the name of the language model backbone (e.g., "LLaMA", "Vicuna", "Qwen2", "InternLM2"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Chat(Tensor<>,IEnumerable<ValueTuple<String,String>>,String)` | Generates a response in a multi-turn chat context with visual input. |

