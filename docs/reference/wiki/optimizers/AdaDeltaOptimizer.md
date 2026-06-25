---
title: "AdaDeltaOptimizer"
description: "Implements the AdaDelta optimization algorithm for training neural networks and other machine learning models."
section: "Reference"
---

_Optimizers_

Implements the AdaDelta optimization algorithm for training neural networks and other machine learning models.

## For Beginners

AdaDelta is like a smart assistant that helps your model learn more efficiently.

Imagine you're learning a new skill:

- Sometimes you need to practice more on difficult parts (bigger learning steps)
- Other times you need to be more careful with easier parts (smaller learning steps)

AdaDelta does this automatically for each part of your model, helping it learn better and faster.
It remembers recent changes and uses this information to decide how big the next learning step should be.

## How It Works

AdaDelta is an adaptive learning rate method that dynamically adjusts the learning rate for each parameter
based on a moving window of gradient updates. This optimizer addresses some of the drawbacks of AdaGrad,
particularly its aggressive, monotonically decreasing learning rate.

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
    .ConfigureOptimizer(new AdaDeltaOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdaDeltaOptimizer.");
```

