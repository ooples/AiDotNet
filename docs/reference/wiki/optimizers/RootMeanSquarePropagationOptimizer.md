---
title: "RootMeanSquarePropagationOptimizer"
description: "Implements the Root Mean Square Propagation (RMSProp) optimization algorithm, an adaptive learning rate method."
section: "Reference"
---

_Optimizers_

Implements the Root Mean Square Propagation (RMSProp) optimization algorithm, an adaptive learning rate method.

## For Beginners

RMSProp is like a hiker who adjusts their step size differently for each direction. Imagine a hiker exploring mountains with different terrains: - On steep slopes (large gradients), the hiker takes small, careful steps - On gentle slopes (small gradients), the hiker takes larger, confident steps - The hiker remembers how steep each direction has been recently (using a moving average) - This memory helps the hiker adjust their steps even as the terrain changes This adaptive approach helps the algorithm find good solutions more quickly by: - Preventing wild overshooting on steep slopes - Making faster progress on gentle terrain - Adjusting automatically to different parts of the solution space

## How It Works

RMSProp is an adaptive learning rate optimization algorithm designed to handle non-stationary objectives and accelerate convergence. It maintains a moving average of the squared gradients for each parameter and divides the learning rate by the square root of this average. This approach allows the algorithm to use a larger learning rate for parameters with small gradients and a smaller learning rate for parameters with large gradients, leading to more efficient optimization.

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
    .ConfigureOptimizer(new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with RootMeanSquarePropagationOptimizer.");
```

