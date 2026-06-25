---
title: "AdaMaxOptimizer"
description: "Represents an AdaMax optimizer, an extension of Adam that uses the infinity norm."
section: "Reference"
---

_Optimizers_

Represents an AdaMax optimizer, an extension of Adam that uses the infinity norm.

## For Beginners

AdaMax is like a smart learning assistant that adjusts its learning speed for each piece of information it's trying to learn. It's particularly good at handling different scales of information without getting confused. Key features: - Adapts the learning rate for each parameter - Uses the maximum (infinity norm) of past gradients, which can be more stable - Good for problems where the gradients can be sparse or have different scales

## How It Works

AdaMax is an adaptive learning rate optimization algorithm that extends the Adam optimizer. It uses the infinity norm to update parameters, which can make it more robust in certain scenarios.

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
    .ConfigureOptimizer(new AdaMaxOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdaMaxOptimizer.");
```

