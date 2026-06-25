---
title: "SEALAdaptiveLearningRateMode"
description: "Specifies the mode for computing adaptive learning rates in SEAL."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Specifies the mode for computing adaptive learning rates in SEAL.

## For Beginners

Different ways to adjust the step size for each parameter:

- GradientNorm: Big gradients get smaller steps (like AdaGrad)
- RunningMean: Uses a moving average of past gradients (like RMSprop)
- PerLayer: All parameters in a layer share the same adaptive rate

## Fields

| Field | Summary |
|:-----|:--------|
| `GradientNorm` | Computes adaptive learning rate based on instantaneous gradient norm. |
| `PerLayer` | Computes one adaptive rate per layer (averaged across layer parameters). |
| `RunningMean` | Uses exponential moving average of squared gradients. |

