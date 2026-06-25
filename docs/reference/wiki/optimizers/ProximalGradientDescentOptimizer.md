---
title: "ProximalGradientDescentOptimizer"
description: "Implements a Proximal Gradient Descent optimization algorithm which combines gradient descent with regularization."
section: "Reference"
---

_Optimizers_

Implements a Proximal Gradient Descent optimization algorithm which combines gradient descent with regularization.

## For Beginners

Proximal Gradient Descent is like walking downhill while staying within certain boundaries.

Imagine you're hiking down a mountain to find the lowest point:

- Standard gradient descent is like always walking directly downhill
- Proximal gradient descent adds boundaries or "guardrails" to your path
- These guardrails keep you from wandering into areas that might look good but are actually not helpful
- For example, the guardrails might prevent solutions that are too complex and would overfit the data

This approach helps find solutions that not only fit the data well but also have desirable properties
like simplicity or stability.

## How It Works

Proximal Gradient Descent (PGD) is an extension of standard gradient descent that handles regularization more
efficiently. The algorithm alternates between performing a gradient descent step to minimize the loss function
and applying a proximal operator to enforce regularization. This approach is particularly effective for
problems where regularization is important to prevent overfitting or to enforce specific properties in the solution.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(0);
var trainX = new Tensor<double>(new[] { 32, 8 });
var trainY = new Tensor<double>(new[] { 32, 2 });
for (int i = 0; i < 32; i++)
{
    for (int j = 0; j < 8; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 2 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 8, numClasses: 2, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new ProximalGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with ProximalGradientDescentOptimizer.");
```

