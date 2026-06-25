---
title: "SkyworkR1V2<T>"
description: "Skywork R1V2: hybrid RL (MPO + GRPO) for multimodal reasoning SOTA."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Reasoning`

Skywork R1V2: hybrid RL (MPO + GRPO) for multimodal reasoning SOTA.

## For Beginners

Skywork R1V2 is an advanced vision-language model using hybrid
reinforcement learning for state-of-the-art multimodal reasoning. Default values follow
the original paper settings.

## How It Works

Skywork R1V2 (2025) achieves state-of-the-art multimodal reasoning through hybrid
reinforcement learning combining MPO (Multi-head Policy Optimization) and GRPO (Group
Relative Policy Optimization) objectives. This hybrid RL approach improves both the
quality of reasoning chains and the accuracy of final answers on complex visual tasks.

**References:**

- Paper: "Skywork R1V2: Multimodal Hybrid Reinforcement Learning" (2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Skywork R1V2's hybrid RL-aligned pipeline. |
| `ReasonWithChainOfThought(Tensor<>,String)` | Generates reasoning using Skywork R1V2's hybrid RL chain-of-thought. |

