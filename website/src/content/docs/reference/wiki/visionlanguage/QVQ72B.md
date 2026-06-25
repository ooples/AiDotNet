---
title: "QVQ72B<T>"
description: "QVQ-72B: first open-source multimodal reasoning model from Qwen."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Reasoning`

QVQ-72B: first open-source multimodal reasoning model from Qwen.

## For Beginners

QVQ-72B is a large open-source multimodal model with visual
chain-of-thought reasoning capabilities. Default values follow the original paper
settings.

## How It Works

QVQ-72B (Qwen Team, 2024) is the first open-source multimodal reasoning model, built
on the Qwen architecture with 72 billion parameters. It features visual chain-of-thought
reasoning where the model generates detailed step-by-step analysis of visual content
before producing answers, achieving strong performance on mathematical and scientific
visual reasoning benchmarks.

**References:**

- Paper: "QVQ: To See the World with Wisdom" (Qwen Team, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using QVQ-72B's Qwen2-VL encoder with visual reasoning. |
| `ReasonWithChainOfThought(Tensor<>,String)` | Generates multi-step reasoning using QVQ-72B's RL-aligned chain-of-thought. |

