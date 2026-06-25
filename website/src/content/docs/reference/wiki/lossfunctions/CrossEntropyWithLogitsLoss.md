---
title: "CrossEntropyWithLogitsLoss<T>"
description: "Implements Cross-Entropy loss that accepts raw logits (not probabilities)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements Cross-Entropy loss that accepts raw logits (not probabilities).

## For Beginners

This loss function is equivalent to PyTorch's
`nn.CrossEntropyLoss`. It combines LogSoftmax and Negative Log-Likelihood
in a single numerically stable computation.

Unlike `CrossEntropyLoss` which expects probability inputs
(after softmax), this version accepts raw logits (unbounded model outputs)
and applies the softmax internally. This is the correct choice when your
model's final layer outputs raw scores without softmax activation.

The formula uses the log-sum-exp trick for numerical stability:
loss = -logit[target_class] + log(sum(exp(logit_i)))

For soft targets (one-hot encoded):
loss = -sum(target_i * (logit_i - log(sum(exp(logit_j)))))

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new CrossEntropyWithLogitsLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"CrossEntropyWithLogitsLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of cross-entropy loss with respect to logits. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Cross-Entropy loss from raw logits using log-sum-exp for stability. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |
| `ResolveClassAxis(Tensor<>,Tensor<>)` | Converts a class-index tensor of shape `S` into a one-hot tensor of shape `[S ; numClasses]` by appending a class axis of length `numClasses` and setting one entry per slot to `1`. |

