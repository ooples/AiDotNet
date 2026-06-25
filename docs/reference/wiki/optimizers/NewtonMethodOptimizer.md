---
title: "NewtonMethodOptimizer"
description: "Implements the Newton's Method optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the Newton's Method optimization algorithm.

## For Beginners

Imagine you're trying to find the lowest point in a valley. Gradient descent is like rolling a ball and letting it follow the slope. Newton's Method is like using a telescope to look at the whole valley, predicting where the lowest point is, and jumping directly there. It's often faster but requires more complex calculations at each step.

## How It Works

Newton's Method is a powerful optimization algorithm that uses both first and second derivatives of the objective function. It often converges faster than first-order methods, especially near the optimum, but can be computationally expensive due to the need to compute and invert the Hessian matrix.

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
    .ConfigureOptimizer(new NewtonMethodOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NewtonMethodOptimizer.");
```

