---
title: "MiniBatchGradientDescentOptimizer"
description: "Implements the Mini-Batch Gradient Descent optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the Mini-Batch Gradient Descent optimization algorithm.

## For Beginners

Imagine you're trying to find the bottom of a valley while blindfolded. Mini-Batch Gradient Descent is like taking a few steps, checking your position, adjusting your direction, and repeating. It's faster than checking after every single step (Stochastic Gradient Descent) but more precise than taking a lot of steps before checking (Batch Gradient Descent).

## How It Works

Mini-Batch Gradient Descent is a variation of gradient descent that splits the training data into small batches to calculate model error and update model coefficients. This approach strikes a balance between the efficiency of stochastic gradient descent and the stability of batch gradient descent.

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
    .ConfigureOptimizer(new MiniBatchGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with MiniBatchGradientDescentOptimizer.");
```

