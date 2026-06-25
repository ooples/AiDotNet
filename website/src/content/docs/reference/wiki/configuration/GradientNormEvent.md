---
title: "GradientNormEvent"
description: "Emitted per parameter tensor after `tape.ComputeGradients` returns, before the optimizer step runs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

Emitted per parameter tensor after `tape.ComputeGradients`
returns, before the optimizer step runs. Captures the L2 norm of the
gradient flowing back into that tensor. The fingerprint of
gradient-flow bugs (issue #1328) is: head-side params show non-zero
norm but embedding / attention QKV / output-projection norms are
zero or near-zero.

## Properties

| Property | Summary |
|:-----|:--------|
| `GradientL2Norm` | L2 norm of the gradient at emission time; 0 when `HasGradient` is false. |
| `HasGradient` | True when the gradient tape produced a gradient for this parameter. |
| `LayerCategory` | Type-safe categorization of the owning layer. |
| `LayerTypeName` | Concrete layer-class name for diagnostic readability. |
| `ParamIndex` | Position of this parameter in the network's enumerated trainable-tensor list. |
| `ParamLength` | Total scalar element count of the parameter tensor. |
| `ParamShape` | Defensive snapshot of the parameter tensor's shape at emission time. |
| `StepIndex` | Sequence number of the training step that emitted this event. |

