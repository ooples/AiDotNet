---
title: "AMSGradOptimizer"
description: "Implements the AMSGrad optimization algorithm, an improved version of Adam optimizer."
section: "Reference"
---

_Optimizers_

Implements the AMSGrad optimization algorithm, an improved version of Adam optimizer.

## For Beginners

AMSGrad is like a smart assistant that helps adjust the learning process. It remembers past information to make better decisions about how quickly to learn in different parts of the problem.

## How It Works

AMSGrad is an adaptive learning rate optimization algorithm that addresses some of the convergence issues in Adam. It maintains the maximum of past squared gradients to ensure non-decreasing step sizes.

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
    .ConfigureOptimizer(new AMSGradOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AMSGradOptimizer.");
```

