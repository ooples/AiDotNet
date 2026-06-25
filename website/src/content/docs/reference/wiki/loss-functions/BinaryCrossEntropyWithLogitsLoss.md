---
title: "BinaryCrossEntropyWithLogitsLoss"
description: "Implements Binary Cross-Entropy loss that accepts raw logits (not probabilities)."
section: "Reference"
---

_Loss Functions_

Implements Binary Cross-Entropy loss that accepts raw logits (not probabilities).

## For Beginners

This loss function is equivalent to PyTorch's
`nn.BCEWithLogitsLoss`. It combines a sigmoid activation and binary
cross-entropy into a single numerically stable computation.

## How It Works

Unlike `BinaryCrossEntropyLoss` which expects probability inputs
(after sigmoid), this version accepts raw logits (unbounded model outputs) and
applies the sigmoid internally. This is the correct choice when your model's
final layer outputs raw scores without sigmoid activation — for example, the
classification heads in DenseNet/EfficientNet/etc. emit logits because applying
sigmoid in the model and then again inside the loss would compose two non-linear
functions and produce wrong gradients.

The numerically stable form (avoids exp overflow for large positive x and
log(0) for very negative x) is:

`loss = max(x, 0) - x * y + log(1 + exp(-|x|))`

Derivation: BCE on probability p = sigmoid(x) is
`-(y log p + (1-y) log(1-p))`. Substituting `p = 1/(1+exp(-x))` and
using `log(1-sigmoid(x)) = -x - log(1+exp(-x))`, the expression simplifies
to `x - x*y + log(1+exp(-x))`. The `max(x, 0) + log(1 + exp(-|x|))`
rewrite is the standard log-sum-exp trick that keeps both the very-positive and
very-negative tails well-conditioned.

The gradient with respect to logits has the elegant form
`d(loss)/d(x) = sigmoid(x) - y`, just like cross-entropy with logits.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new BinaryCrossEntropyWithLogitsLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"BinaryCrossEntropyWithLogitsLoss = {value:F4}");
```

