---
title: "IReasoningVLM<T>"
description: "Interface for vision-language models with chain-of-thought reasoning capabilities."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for vision-language models with chain-of-thought reasoning capabilities.

## How It Works

Reasoning VLMs extend instruction-tuned VLMs with explicit thinking/reasoning steps
before producing a final answer. They are trained with reinforcement learning and/or
chain-of-thought supervision to decompose complex visual reasoning tasks.
Architectures include:

- QVQ/Kimi-VL: MoE language models with visual reasoning alignment
- Skywork R1V: Cross-modal transfer of text reasoning to vision
- LLaVA-CoT: Chain-of-thought fine-tuning on visual instruction data

## Properties

| Property | Summary |
|:-----|:--------|
| `ReasoningApproach` | Gets the name of the reasoning approach (e.g., "CoT", "RL-Aligned", "MoE-Reasoning"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ReasonWithChainOfThought(Tensor<>,String)` | Generates a response with explicit chain-of-thought reasoning steps. |

