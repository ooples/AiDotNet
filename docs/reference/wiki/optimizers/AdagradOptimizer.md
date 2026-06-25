---
title: "AdagradOptimizer"
description: "Represents an Adagrad (Adaptive Gradient) optimizer for gradient-based optimization."
section: "Reference"
---

_Optimizers_

Represents an Adagrad (Adaptive Gradient) optimizer for gradient-based optimization.

## For Beginners

Adagrad is like a smart learning assistant that adjusts how much it learns for each piece of information based on how often it has seen similar information before. - It learns more from new or rare information - It learns less from common or frequently seen information - This helps it focus on the most important parts of what it's learning This can be especially useful when some parts of your data are more important or occur less frequently.

## How It Works

The Adagrad optimizer adapts the learning rate for each parameter based on the historical gradients. It performs larger updates for infrequent parameters and smaller updates for frequent ones.

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
    .ConfigureOptimizer(new AdagradOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdagradOptimizer.");
```

