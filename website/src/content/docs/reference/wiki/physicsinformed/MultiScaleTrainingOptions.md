---
title: "MultiScaleTrainingOptions<T>"
description: "Configuration options for multi-scale PINN training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.PhysicsInformed.Interfaces`

Configuration options for multi-scale PINN training.

## Properties

| Property | Summary |
|:-----|:--------|
| `CouplingWeight` | Coupling loss weight (balances scale coupling vs individual scale losses). |
| `ManualScaleWeights` | Individual weights for each scale (overrides automatic weighting). |
| `ScalePretrainingEpochs` | Number of epochs to pre-train each scale before adding the next. |
| `UseAdaptiveScaleWeighting` | Whether to use adaptive scale weighting during training. |
| `UseSequentialScaleTraining` | Whether to train scales sequentially (coarse to fine) or simultaneously. |

