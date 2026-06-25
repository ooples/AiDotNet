---
title: "LevenbergMarquardtOptimizer"
description: "Implements the Levenberg-Marquardt optimization algorithm for non-linear least squares problems."
section: "Reference"
---

_Optimizers_

Implements the Levenberg-Marquardt optimization algorithm for non-linear least squares problems.

## For Beginners

This optimizer is like a smart problem-solver that's really good at fitting curves to data points. It's especially useful when the relationship between your inputs and outputs isn't a straight line. It works by making small adjustments to its guess, getting closer to the best solution with each step.

## How It Works

The Levenberg-Marquardt algorithm is a popular method for solving non-linear least squares problems. It combines the Gauss-Newton algorithm and the method of gradient descent, providing a robust solution that works well in many situations.

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
    .ConfigureOptimizer(new LevenbergMarquardtOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LevenbergMarquardtOptimizer.");
```

