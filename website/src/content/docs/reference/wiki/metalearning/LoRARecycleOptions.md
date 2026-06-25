---
title: "LoRARecycleOptions<T, TInput, TOutput>"
description: "Configuration options for LoRA-Recycle (Hu et al., CVPR 2025)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for LoRA-Recycle (Hu et al., CVPR 2025).

## How It Works

LoRA-Recycle distills a "meta-LoRA" from diverse pre-tuned LoRA adapters without
accessing their private training data. It supports tuning-free few-shot adaptation
by recycling knowledge from previously-learned tasks via prototype-based matching.

**Key Parameters:**

- `NumRecycledAdapters` — number of previously-learned LoRA adapters to maintain
- `Rank` — rank of each LoRA adapter
- `PrototypeDim` — dimensionality of prototype embeddings for adapter selection

## Properties

| Property | Summary |
|:-----|:--------|
| `KLWeight` | KL divergence weight for distillation loss. |
| `NumRecycledAdapters` | Number of recycled LoRA adapters to maintain in the adapter bank. |
| `PrototypeDim` | Dimensionality of the prototype embedding used for adapter selection. |
| `Rank` | Rank of each LoRA adapter (controls adapter capacity). |
| `SelectionTemperature` | Temperature parameter for softmax-based adapter weighting. |

