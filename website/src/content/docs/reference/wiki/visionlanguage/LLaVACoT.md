---
title: "LLaVACoT<T>"
description: "LLaVA-CoT: chain-of-thought visual reasoning with structured output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Reasoning`

LLaVA-CoT: chain-of-thought visual reasoning with structured output.

## For Beginners

LLaVA-CoT is a vision-language model that reasons step-by-step
through visual problems using chain-of-thought. Default values follow the original paper
settings.

## How It Works

LLaVA-CoT (2024) extends the LLaVA architecture with chain-of-thought visual reasoning,
generating structured step-by-step reasoning traces for complex visual tasks. The model
produces intermediate reasoning steps including observation, hypothesis, and conclusion
phases before arriving at the final answer, improving accuracy on visual reasoning benchmarks.

**References:**

- Paper: "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using LLaVA-CoT's structured reasoning pipeline. |
| `ReasonWithChainOfThought(Tensor<>,String)` | Generates structured chain-of-thought reasoning using LLaVA-CoT's 4-stage pipeline. |

