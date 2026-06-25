---
title: "IFusedActivation"
description: "Implemented by scalar activation functions that have an exact fused-kernel equivalent (`FusedActivationType`), so fused inference paths such as `IEngine.MlpForward` can ask the activation what kernel it maps to instead of switching on its t…"
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActivationFunctions.Fused`

Implemented by scalar activation functions that have an exact fused-kernel
equivalent (`FusedActivationType`), so fused inference paths such
as `IEngine.MlpForward` can ask the activation what kernel it maps to
instead of switching on its type.

## How It Works

Open/closed-compliant for the same reason as
`IFusedOptimizerSpec`: only activations whose
fused kernel is numerically identical to the scalar form implement this, so
there is no central activation→enum switch to maintain and an unrecognized
activation simply keeps the generic per-layer path. A null activation (linear
layer) is treated as `None` by callers.

`FusedActivationType@)` returns `false` when THIS instance
can't be reproduced by the fused kernel — e.g. a parametric activation whose
parameter (LeakyReLU slope, ELU alpha) differs from the value the kernel
hardcodes — so a custom-parameter instance correctly falls back rather than
silently getting the kernel's default parameter.

## Methods

| Method | Summary |
|:-----|:--------|
| `TryGetFusedActivation(FusedActivationType)` | Reports the fused-kernel activation type equivalent to this activation, or returns `false` if this instance can't be reproduced by the kernel. |

