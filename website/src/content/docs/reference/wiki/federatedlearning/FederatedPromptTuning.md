---
title: "FederatedPromptTuning<T>"
description: "Federated Prompt Tuning — soft prompt aggregation for foundation model personalization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Federated Prompt Tuning — soft prompt aggregation for foundation model personalization.

## For Beginners

Instead of modifying the model itself, prompt tuning adds a few
learned "instructions" in front of each input. These instructions tell the model how
to adapt to the specific task. In federated prompt tuning, each device learns its own
instructions and they're combined at the server — only a few thousand parameters need
to be shared, even for billion-parameter models.

## How It Works

Prompt tuning (Lester et al., 2021) prepends learnable "soft prompt" tokens to the input.
In federated settings, only these prompt embeddings (typically 10-100 tokens × embedding dim)
are communicated, offering even higher compression than LoRA for very large models.

References:
Lester et al. (2021), "The Power of Scale for Parameter-Efficient Prompt Tuning".
Zhao et al. (2024), "FedPSF-LLM: Dual Prompt Personalization for Federated Foundation Models".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedPromptTuning(Int32,Int32,Int32)` | Creates a new federated prompt tuning strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `CompressionRatio` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ExtractAdapterParameters(Vector<>)` |  |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |

