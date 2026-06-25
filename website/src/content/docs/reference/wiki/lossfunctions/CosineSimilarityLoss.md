---
title: "CosineSimilarityLoss<T>"
description: "Implements the Cosine Similarity Loss between two vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Cosine Similarity Loss between two vectors.

## For Beginners

Cosine Similarity measures how similar two vectors are in terms of their orientation,
regardless of their magnitude (size).

The formula for cosine similarity is: cos(θ) = (A·B)/(||A||×||B||)
Where:

- A·B is the dot product of vectors A and B
- ||A|| and ||B|| are the magnitudes (lengths) of vectors A and B
- θ is the angle between vectors A and B

The loss is calculated as 1 - cosine similarity, so:

- A value of 0 means the vectors are perfectly aligned (very similar)
- A value of 1 means they are perpendicular (no similarity)
- A value of 2 means they point in exactly opposite directions

Cosine similarity loss is particularly useful for:

- Text similarity tasks (comparing document vectors)
- Recommendation systems
- Image retrieval
- Any task where the direction of vectors matters more than their magnitude

It's often preferred over Euclidean distance when working with high-dimensional sparse vectors.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new CosineSimilarityLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"CosineSimilarityLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosineSimilarityLoss` | Initializes a new instance of the CosineSimilarityLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Cosine Similarity Loss with respect to the predicted values. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Cosine Similarity Loss between two vectors. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

