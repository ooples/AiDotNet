---
title: "ContextMetaRLOptions<T, TInput, TOutput>"
description: "Configuration options for Context Meta-RL: context-conditioned meta-reinforcement learning with attention-based aggregation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Context Meta-RL: context-conditioned meta-reinforcement
learning with attention-based aggregation.

## How It Works

Context Meta-RL aggregates task context from support interactions using an
attention mechanism and uses the resulting context vector to modulate the
policy network parameters. Unlike PEARL's Gaussian posterior, this approach
uses deterministic attention-based aggregation with a learned query vector.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionTemperature` | Attention temperature for softmax. |
| `ContextDim` | Dimensionality of the context embedding. |
| `ModulationStrength` | Modulation strength for context-conditioned parameter adjustment. |
| `NumAttentionHeads` | Number of attention heads for context aggregation. |

