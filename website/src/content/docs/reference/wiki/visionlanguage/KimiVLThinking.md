---
title: "KimiVLThinking<T>"
description: "Kimi-VL-Thinking: long chain-of-thought reasoning with RL alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Reasoning`

Kimi-VL-Thinking: long chain-of-thought reasoning with RL alignment.

## For Beginners

Kimi-VL-Thinking is a vision-language model specialized in
long chain-of-thought reasoning for complex visual tasks. Default values follow the
original paper settings.

## How It Works

Kimi-VL-Thinking (Moonshot AI, 2025) extends Kimi-VL with long chain-of-thought reasoning
capabilities trained through reinforcement learning alignment. It generates structured
multi-step reasoning traces for complex visual problems, breaking down tasks into
intermediate steps before producing final answers.

**References:**

- Paper: "Kimi-VL Technical Report" (Moonshot AI, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Kimi-VL-Thinking's RL-aligned long thinking pipeline. |
| `ReasonWithChainOfThought(Tensor<>,String)` | Generates extended reasoning using Kimi-VL-Thinking's RL-aligned long thinking chains. |

