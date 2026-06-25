---
title: "GpuOptimizerState"
description: "Holds the optimizer state buffers for GPU-resident training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Holds the optimizer state buffers for GPU-resident training.

## How It Works

Different optimizers require different state:

- SGD with momentum: velocity buffer
- Adam/AdamW: first moment (m) and second moment (v) buffers
- RMSprop: squared average buffer
- Adagrad: accumulated gradient buffer

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatedGrad` | Accumulated gradient buffer (for Adagrad). |
| `M` | First moment buffer (for Adam-family optimizers). |
| `SquaredAvg` | Squared average buffer (for RMSprop). |
| `V` | Second moment buffer (for Adam-family optimizers). |
| `Velocity` | Velocity buffer (for SGD momentum, NAG, LARS). |

