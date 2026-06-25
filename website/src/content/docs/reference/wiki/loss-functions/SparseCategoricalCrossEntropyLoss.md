---
title: "SparseCategoricalCrossEntropyLoss"
description: "Implements the Sparse Categorical Cross Entropy loss function for multi-class classification with integer labels."
section: "Reference"
---

_Loss Functions_

Implements the Sparse Categorical Cross Entropy loss function for multi-class classification with integer labels.

## For Beginners

Sparse Categorical Cross Entropy is similar to Categorical Cross Entropy but is used
when labels are provided as class indices (0, 1, 2, ...) rather than one-hot encoded vectors.

This is more memory efficient for problems with many classes, as you only need to store the class index
instead of a full one-hot encoded vector.

The formula is: SCCE = -(1/n) * Σ[log(predicted[actual_class_index])]

Where:

- actual contains the class indices (e.g., 0, 1, 2, 3 for a 4-class problem)
- predicted contains the predicted probabilities for all classes
- We extract the probability for the correct class using the index from actual

Example:

- If actual[i] = 2.0 (class index 2), and predicted has probabilities [0.1, 0.2, 0.6, 0.1],

then we take predicted[2] = 0.6 and compute -log(0.6)

Key properties:

- More memory efficient than categorical cross-entropy for many-class problems
- Predicted values should be probabilities (between 0 and 1) from a softmax layer
- Actual values should be valid class indices (0 to num_classes-1)
- Often used with the softmax activation function in neural networks

To use this loss function with the Vector interface:

- For a single sample: predicted = [p_class0, p_class1, ..., p_classN], actual = [true_class_index]
- For batches: flatten your data appropriately or process samples individually

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new SparseCategoricalCrossEntropyLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"SparseCategoricalCrossEntropyLoss = {value:F4}");
```

