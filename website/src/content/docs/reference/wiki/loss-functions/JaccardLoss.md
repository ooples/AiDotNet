---
title: "JaccardLoss"
description: "Implements the Jaccard loss function, commonly used for measuring dissimilarity between sets."
section: "Reference"
---

_Loss Functions_

Implements the Jaccard loss function, commonly used for measuring dissimilarity between sets.

## For Beginners

Jaccard loss measures how dissimilar two sets are. It's calculated as 1 minus 
the size of the intersection divided by the size of the union.

The formula is: 1 - |A n B| / |A ? B|
Where:

- A n B is the intersection of sets A and B (elements in both)
- A ? B is the union of sets A and B (elements in either)

For continuous values (like probabilities), the intersection is the sum of the minimum values,
and the union is the sum of the maximum values at each position.

Key properties:

- A value of 0 means perfect overlap (identical sets)
- A value of 1 means no overlap at all
- It's symmetric: Jaccard(A,B) = Jaccard(B,A)
- It's a proper distance metric, suitable for measuring dissimilarity

Jaccard loss is particularly useful for:

- Image segmentation tasks
- Set similarity problems
- Binary classification problems
- Tasks where the positive class is rare (imbalanced data)

It's often a better choice than pixel-wise losses (like MSE) for segmentation tasks 
because it directly optimizes for the overlap of segments.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new JaccardLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"JaccardLoss = {value:F4}");
```

