---
title: "NoiseContrastiveEstimationLoss"
description: "Implements the Noise Contrastive Estimation (NCE) loss function for efficient training with large output spaces."
section: "Reference"
---

_Loss Functions_

Implements the Noise Contrastive Estimation (NCE) loss function for efficient training with large output spaces.

## For Beginners

Noise Contrastive Estimation (NCE) is a loss function designed to efficiently train models with very large output spaces, such as language models with large vocabularies. Instead of computing probabilities for all possible outputs (which could be millions in language models), NCE transforms the problem into a binary classification task: distinguishing the true data from noise samples. The key idea is to: - Sample a small number of "negative" examples from a noise distribution - Train the model to distinguish between true data points and these negative samples This approach is much more computationally efficient than computing full softmax probabilities over all possible outputs, especially when the output space is very large. NCE is commonly used in: - Word embedding models like Word2Vec - Neural language models with large vocabularies - Any model with a very large output space

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new NoiseContrastiveEstimationLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"NoiseContrastiveEstimationLoss = {value:F4}");
```

