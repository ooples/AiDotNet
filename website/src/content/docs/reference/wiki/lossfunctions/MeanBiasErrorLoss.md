---
title: "MeanBiasErrorLoss<T>"
description: "Implements the Mean Bias Error (MBE) loss function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Mean Bias Error (MBE) loss function.

## For Beginners

Mean Bias Error is a diagnostic metric that reveals systematic bias in predictions.
Unlike MSE or MAE which measure error magnitude, MBE tells you the *direction* of errors.

The formula is: MBE = (1/n) * Σ(actual - predicted)

Think of MBE like this:

- MBE = 0: Your model is unbiased (errors cancel out)
- MBE > 0: Your model tends to under-predict (predictions are too low)
- MBE < 0: Your model tends to over-predict (predictions are too high)

Example scenarios:

- Weather forecasting: MBE = +2°C means you're consistently predicting 2 degrees too cold
- Price predictions: MBE = -$5,000 means you're consistently overestimating by $5,000
- Medical diagnostics: MBE helps detect if a test systematically over/under-estimates values

Key properties:

- Can be positive, negative, or zero (unlike MSE/RMSE which are always non-negative)
- Errors of opposite signs cancel each other out
- Not sensitive to the magnitude of individual errors
- Useful for detecting systematic bias, not for measuring overall accuracy
- Often used alongside RMSE or MAE for a complete error analysis

MBE is ideal for:

- Diagnosing systematic prediction bias
- Calibrating models that consistently over/under-predict
- Quality control in measurement systems
- Understanding if your model needs adjustment in a specific direction

**Important:** MBE should not be used alone for model evaluation, as positive and negative
errors cancel out. A model with large errors in both directions could have MBE ≈ 0.
Always use MBE together with metrics like RMSE or MAE.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new MeanBiasErrorLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"MeanBiasErrorLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Mean Bias Error loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Mean Bias Error between predicted and actual values. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

