---
title: "KullbackLeiblerDivergence"
description: "Implements the Kullback-Leibler Divergence, a measure of how one probability distribution differs from another."
section: "Reference"
---

_Loss Functions_

Implements the Kullback-Leibler Divergence, a measure of how one probability distribution differs from another.

## For Beginners

Kullback-Leibler (KL) Divergence measures how one probability distribution differs from another.
It's often interpreted as the "information loss" when using one distribution to approximate another.

The formula is: KL(P||Q) = sum(P(i) * log(P(i)/Q(i))
Where:

- P is the true distribution
- Q is the approximating distribution

Key properties:

- It's always non-negative (zero only when the distributions are identical)
- It's not symmetric: KL(P||Q) ? KL(Q||P)
- It's not a true distance metric due to this asymmetry

KL divergence is commonly used in:

- Variational Autoencoders (VAEs)
- Reinforcement learning algorithms
- Information theory applications
- Distribution approximation tasks

When training models, KL divergence helps push the predicted distribution (Q) to match the target distribution (P).

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new KullbackLeiblerDivergence<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"KullbackLeiblerDivergence = {value:F4}");
```

