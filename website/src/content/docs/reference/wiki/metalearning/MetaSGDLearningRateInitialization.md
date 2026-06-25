---
title: "MetaSGDLearningRateInitialization"
description: "Learning rate initialization strategies for Meta-SGD."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Learning rate initialization strategies for Meta-SGD.

## How It Works

Different initialization strategies can affect how quickly Meta-SGD converges
and the quality of the learned per-parameter optimizers.

## Fields

| Field | Summary |
|:-----|:--------|
| `LayerBased` | Initialize learning rates based on layer depth. |
| `MagnitudeBased` | Initialize learning rates based on parameter magnitudes. |
| `Random` | Initialize learning rates uniformly at random within LearningRateInitRange. |
| `Uniform` | Initialize all learning rates to the same value (InnerLearningRate). |
| `Xavier` | Initialize learning rates using Xavier-style initialization. |

