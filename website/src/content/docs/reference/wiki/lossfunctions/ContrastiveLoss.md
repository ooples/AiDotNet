---
title: "ContrastiveLoss<T>"
description: "Implements the Contrastive Loss function for learning similarity metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Contrastive Loss function for learning similarity metrics.

## For Beginners

Contrastive Loss helps a model learn to identify whether two examples are similar or dissimilar.
It works with pairs of examples and their similarity label (1 for similar, 0 for dissimilar).

For similar pairs, the loss penalizes distance between them, encouraging them to be close together.
For dissimilar pairs, the loss penalizes proximity below a certain margin, encouraging them to be at least
that far apart.

The formula has two components:

- For similar pairs (y=1): distance²
- For dissimilar pairs (y=0): max(0, margin - distance)²

Contrastive Loss is commonly used in:

- Siamese neural networks
- Face verification systems (determining if two faces are the same person)
- Signature verification
- Any situation where you need to learn a similarity metric between pairs

This approach is simpler than Triplet Loss as it only requires pairs of examples rather than triplets.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new ContrastiveLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"ContrastiveLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContrastiveLoss(Double)` | Initializes a new instance of the ContrastiveLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | This method is not used for Contrastive Loss as it requires two input vectors and a similarity label. |
| `CalculateDerivative(Vector<>,Vector<>,)` | Calculates the gradients of the Contrastive Loss function for both output vectors. |
| `CalculateLoss(Vector<>,Vector<>)` | This method is not used for Contrastive Loss as it requires two input vectors and a similarity label. |
| `CalculateLoss(Vector<>,Vector<>,)` | Calculates the Contrastive Loss between two output vectors based on their similarity. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>,Tensor<>)` | Calculates Contrastive Loss on GPU for batched input tensors. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_margin` | The margin that enforces separation between dissimilar pairs. |

