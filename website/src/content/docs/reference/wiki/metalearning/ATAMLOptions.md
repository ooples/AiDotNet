---
title: "ATAMLOptions<T, TInput, TOutput>"
description: "Configuration options for the ATAML (Attention-based Task-Adaptive Meta-Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the ATAML (Attention-based Task-Adaptive Meta-Learning) algorithm.

## How It Works

ATAML learns per-parameter attention weights that modulate the inner learning rate based
on the task's gradient profile. A small attention network maps compressed gradient features
to per-parameter scaling factors via softmax, allowing task-adaptive parameter updates.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDim` | Dimension of the compressed gradient representation used as input to the attention network. |
| `AttentionEntropyWeight` | Entropy regularization weight on the attention distribution to prevent collapse. |
| `AttentionTemperature` | Temperature for the attention softmax. |

