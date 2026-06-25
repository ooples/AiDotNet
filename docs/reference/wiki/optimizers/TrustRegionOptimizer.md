---
title: "TrustRegionOptimizer"
description: "Implements the Trust Region optimization algorithm for machine learning models."
section: "Reference"
---

_Optimizers_

Implements the Trust Region optimization algorithm for machine learning models.

## For Beginners

Think of this optimizer as an explorer with a map: - The "trust region" is like the area on the map the explorer trusts to be accurate. - In each step, the explorer looks at this trusted area to decide where to go next. - If the predictions (map) match reality well, the explorer might expand the trusted area. - If the predictions are off, the explorer shrinks the trusted area and becomes more cautious. This approach helps the optimizer make good decisions even in complex landscapes, balancing between making progress and staying reliable.

## How It Works

The Trust Region optimizer is an advanced optimization technique that uses local quadratic approximations of the objective function to determine the next step. It maintains a region of trust around the current solution where the approximation is considered reliable.

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
    .ConfigureOptimizer(new TrustRegionOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with TrustRegionOptimizer.");
```

