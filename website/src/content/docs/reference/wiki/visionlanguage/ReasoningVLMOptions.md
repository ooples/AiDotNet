---
title: "ReasoningVLMOptions"
description: "Base configuration options for reasoning vision-language models with chain-of-thought capabilities."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Reasoning`

Base configuration options for reasoning vision-language models with chain-of-thought capabilities.

## For Beginners

These options configure the Reasoning model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReasoningVLMOptions` | Initializes a new instance with default values. |
| `ReasoningVLMOptions(ReasoningVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableThinkingSteps` | Gets or sets whether to include explicit thinking steps in the output. |
| `MaxReasoningTokens` | Gets or sets the maximum number of reasoning tokens before the final answer. |
| `ReasoningApproach` | Gets or sets the reasoning approach (e.g., "CoT", "RL-Aligned", "MoE-Reasoning"). |

