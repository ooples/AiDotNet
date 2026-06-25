---
title: "BornRuleMseLoss<T>"
description: "Mean-squared error measured in *probability* space for models whose final layer emits quantum *amplitudes*: `loss = mean((predicted² − target)²)`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Mean-squared error measured in *probability* space for models whose final
layer emits quantum *amplitudes*: `loss = mean((predicted² − target)²)`.

## How It Works

A quantum neural network's layer chain produces amplitudes; the observable
(Born's rule) is the squared magnitude, `p = amplitude²`. Training such a
model on probability targets with a plain MSE on the amplitudes optimises the
wrong objective — minimising `‖amplitude − √target‖²` does NOT monotonically
minimise the measured `‖amplitude² − target‖²` because the square is
non-linear. This loss folds the Born-rule square into the objective so the tape
trains exactly the measured quantity the model reports at inference, keeping the
training loss and the measured (probability-space) error in lock-step.

Gradient w.r.t. the amplitude prediction `a`:
`d/da mean((a² − t)²) = (4/n) · a · (a² − t)`.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new BornRuleMseLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"BornRuleMseLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` |  |
| `CalculateLoss(Vector<>,Vector<>)` |  |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

