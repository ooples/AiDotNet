---
title: "ElasticNetLoss<T>"
description: "Implements the Elastic Net Loss function, which combines Mean Squared Error with L1 and L2 regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Elastic Net Loss function, which combines Mean Squared Error with L1 and L2 regularization.

## For Beginners

Elastic Net Loss combines the Mean Squared Error (which measures prediction accuracy) 
with two types of regularization (which prevent overfitting):

- L1 regularization (also called Lasso): Helps select only the most important features by pushing some weights to zero
- L2 regularization (also called Ridge): Prevents any single weight from becoming too large

The formula is: MSE + a * [l1Ratio * |weights|_1 + (1-l1Ratio) * 0.5 * |weights|_2²]
Where:

- MSE is the Mean Squared Error
- |weights|_1 is the L1 norm (sum of absolute values)
- |weights|_2² is the squared L2 norm (sum of squared values)
- a is the regularization strength
- l1Ratio controls the mix between L1 and L2 regularization

The l1Ratio parameter (between 0 and 1) controls the balance:

- When l1Ratio = 1: Only L1 regularization is used (Lasso)
- When l1Ratio = 0: Only L2 regularization is used (Ridge)
- Values in between: A mix of both (Elastic Net)

This loss function is particularly useful when:

- You have many correlated features
- You want to perform feature selection (L1 component)
- You still want the stability of L2 regularization
- You want to balance between model simplicity and prediction accuracy

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new ElasticNetLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"ElasticNetLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticNetLoss(Double,Double)` | Initializes a new instance of the ElasticNetLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Elastic Net Loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Elastic Net Loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Elastic Net loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |
| `SignOf()` | Returns the sign of a value: -1 for negative, 1 for positive, 0 for zero. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The regularization strength parameter. |
| `_l1Ratio` | The mixing parameter between L1 and L2 regularization. |

