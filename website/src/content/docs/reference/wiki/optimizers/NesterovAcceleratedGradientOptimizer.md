---
title: "NesterovAcceleratedGradientOptimizer"
description: "Implements the Nesterov Accelerated Gradient optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the Nesterov Accelerated Gradient optimization algorithm.

## For Beginners

Imagine you're skiing down a hill. Regular gradient descent is like looking at your current position to decide where to go next.
NAG is like looking ahead to where you'll be after your next move, and then deciding how to adjust your path.
This "look-ahead" helps you navigate the slope more efficiently, especially around tricky turns.

## How It Works

The Nesterov Accelerated Gradient (NAG) is an optimization algorithm that improves upon standard gradient descent.
It introduces a smart prediction of the next position of the parameters, which helps to dampen oscillations and
improve convergence, especially in scenarios with high curvature or small but consistent gradients.

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
    .ConfigureOptimizer(new NesterovAcceleratedGradientOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NesterovAcceleratedGradientOptimizer.");
```

