---
title: "FTRLOptimizer"
description: "Represents a Follow The Regularized Leader (FTRL) optimizer for machine learning models."
section: "Reference"
---

_Optimizers_

Represents a Follow The Regularized Leader (FTRL) optimizer for machine learning models.

## For Beginners

FTRL is an advanced optimization technique that's good at handling large-scale, sparse data. It's often used in online advertising and recommendation systems. Think of FTRL like a smart learning system that: - Adjusts its learning speed for each feature independently - Can handle situations where most features are zero (sparse data) - Is good at balancing between finding a good solution and not overfitting For example, in an online advertising system, FTRL might: - Quickly learn which ad categories are important for a user - Ignore or learn slowly from features that rarely appear - Automatically adjust how much it learns from each new piece of data This makes FTRL particularly good for systems that need to learn and predict in real-time with lots of data.

## How It Works

The FTRL optimizer is an online learning algorithm that adapts regularization in a per-coordinate fashion. It's particularly effective for sparse datasets and is widely used in click-through rate (CTR) prediction and other online learning scenarios.

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
    .ConfigureOptimizer(new FTRLOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with FTRLOptimizer.");
```

