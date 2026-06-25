---
title: "QuantileLoss<T>"
description: "Implements the Quantile loss function for quantile regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Quantile loss function for quantile regression.

## For Beginners

Quantile loss helps predict specific percentiles of data rather than just the average.

For example:

- With quantile=0.5, it predicts the median value (50th percentile)
- With quantile=0.9, it predicts the 90th percentile
- With quantile=0.1, it predicts the 10th percentile

This is useful when you care more about certain parts of the distribution, such as:

- Predicting worst-case scenarios (high quantiles)
- Ensuring predictions don't fall below a certain threshold (low quantiles)
- Creating prediction intervals (by predicting multiple quantiles)

The loss function applies different penalties for overestimation versus underestimation,
which forces the model to learn the specific quantile rather than just the average.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new QuantileLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"QuantileLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuantileLoss(Double)` | Initializes a new instance of the QuantileLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Quantile loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Quantile loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Quantile loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_quantile` | The quantile value to estimate (between 0 and 1). |

