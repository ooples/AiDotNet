---
title: "TemperatureScaling<T>"
description: "TemperatureScaling<T> — Models & Types in AiDotNet.UncertaintyQuantification.Calibration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.UncertaintyQuantification.Calibration`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemperatureScaling(Double)` | Initializes a new instance of the TemperatureScaling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Temperature` | Gets or sets the temperature parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(Matrix<>,Vector<Int32>,Double,Int32)` | Calibrates the temperature parameter using validation data. |
| `ComputeGradient(Matrix<>,Vector<Int32>)` | Computes the gradient of negative log-likelihood with respect to temperature. |
| `ScaleLogits(Tensor<>)` | Applies temperature scaling to logits. |
| `Softmax(Vector<>)` | Computes softmax probabilities from logits. |

