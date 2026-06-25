---
title: "LionOptimizer"
description: "Implements the Lion (Evolved Sign Momentum) optimization algorithm for gradient-based optimization."
section: "Reference"
---

_Optimizers_

Implements the Lion (Evolved Sign Momentum) optimization algorithm for gradient-based optimization.

## For Beginners

Lion is like a simplified but more powerful version of Adam. Instead of carefully measuring how big each step should be (like Adam does), Lion only looks at which direction to go and takes consistent-sized steps in that direction. This is like following a compass that only shows direction - it's simpler, uses less memory, and often gets you to your destination faster. Lion is particularly good for training large neural networks.

## How It Works

Lion is a modern optimization algorithm discovered through symbolic program search that offers significant advantages over traditional optimizers like Adam. It achieves 50% memory reduction by maintaining only a single momentum state (compared to Adam's two states) while often achieving superior performance on large transformer models and other deep learning architectures. 

The algorithm uses sign-based gradient updates, which provides implicit regularization and better generalization. Unlike Adam's magnitude-based updates, Lion focuses purely on the direction of gradients, making it more robust to gradient scale variations and leading to more consistent training dynamics.

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
    .ConfigureOptimizer(new LionOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LionOptimizer.");
```

