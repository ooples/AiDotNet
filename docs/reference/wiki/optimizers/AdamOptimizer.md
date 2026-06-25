---
title: "AdamOptimizer"
description: "Implements the Adam (Adaptive Moment Estimation) optimization algorithm for gradient-based optimization."
section: "Reference"
---

_Optimizers_

Implements the Adam (Adaptive Moment Estimation) optimization algorithm for gradient-based optimization.

## For Beginners

Adam is like a smart personal trainer for your machine learning model.
It helps your model learn efficiently by adjusting how it learns based on past experiences.

## How It Works

Adam is an advanced optimization algorithm that combines ideas from RMSprop and Momentum optimization methods.
It adapts the learning rates for each parameter individually and is well-suited for problems with noisy or sparse gradients.

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
    .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdamOptimizer.");
```

