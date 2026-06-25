---
title: "ContrastiveLoss"
description: "Implements the Contrastive Loss function for learning similarity metrics."
section: "Reference"
---

_Loss Functions_

Implements the Contrastive Loss function for learning similarity metrics.

## For Beginners

Contrastive Loss helps a model learn to identify whether two examples are similar or dissimilar. It works with pairs of examples and their similarity label (1 for similar, 0 for dissimilar). For similar pairs, the loss penalizes distance between them, encouraging them to be close together. For dissimilar pairs, the loss penalizes proximity below a certain margin, encouraging them to be at least that far apart. The formula has two components: - For similar pairs (y=1): distance² - For dissimilar pairs (y=0): max(0, margin - distance)² Contrastive Loss is commonly used in: - Siamese neural networks - Face verification systems (determining if two faces are the same person) - Signature verification - Any situation where you need to learn a similarity metric between pairs This approach is simpler than Triplet Loss as it only requires pairs of examples rather than triplets.

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

