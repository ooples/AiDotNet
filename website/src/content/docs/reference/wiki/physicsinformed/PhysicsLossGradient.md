---
title: "PhysicsLossGradient<T>"
description: "Holds loss and gradient information for physics-informed objectives."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed`

Holds loss and gradient information for physics-informed objectives.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PhysicsLossGradient(Int32,Int32,INumericOperations<>)` | Initializes a new instance of the `PhysicsLossGradient` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FirstDerivatives` | Gradient of loss with respect to first derivatives. |
| `Loss` | Gets or sets the loss value for this sample. |
| `OutputGradients` | Gradient of loss with respect to outputs. |
| `SecondDerivatives` | Gradient of loss with respect to second derivatives. |

