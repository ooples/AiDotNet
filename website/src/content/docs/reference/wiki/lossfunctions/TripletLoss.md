---
title: "TripletLoss<T>"
description: "Implements the Triplet Loss function for learning similarity embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Triplet Loss function for learning similarity embeddings.

## For Beginners

Triplet Loss helps create embeddings (numerical representations) where similar items 
are close together and different items are far apart in a vector space.

It works with triplets of data:

- Anchor: A reference point (e.g., a person's face)
- Positive: An example similar to the anchor (e.g., another image of the same person)
- Negative: An example different from the anchor (e.g., an image of a different person)

The loss encourages the model to make the distance between the anchor and positive smaller than
the distance between the anchor and negative by at least a specified margin.

This loss function is commonly used in:

- Face recognition systems
- Image retrieval applications
- Recommendation systems
- Any task where you need to learn meaningful similarity metrics

By minimizing triplet loss, the model learns to create an embedding space where semantically 
similar items cluster together and dissimilar items are pushed apart.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new TripletLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"TripletLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TripletLoss(Double)` | Initializes a new instance of the TripletLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Matrix<>,Matrix<>,Matrix<>)` | Calculates the gradients of the Triplet Loss function for anchor, positive, and negative samples. |
| `CalculateDerivative(Vector<>,Vector<>)` | This method is not used for Triplet Loss as it requires multiple input vectors. |
| `CalculateLoss(Matrix<>,Matrix<>,Matrix<>)` | Calculates the Triplet Loss for embedding learning. |
| `CalculateLoss(Vector<>,Vector<>)` | This method is not used for Triplet Loss as it requires multiple input vectors. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>,Tensor<>)` | Calculates Triplet Loss on GPU for batched input tensors. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_margin` | The margin that enforces separation between positive and negative pairs. |

