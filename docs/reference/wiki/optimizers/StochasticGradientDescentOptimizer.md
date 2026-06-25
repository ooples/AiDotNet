---
title: "StochasticGradientDescentOptimizer"
description: "Represents a Stochastic Gradient Descent (SGD) optimizer for machine learning models."
section: "Reference"
---

_Optimizers_

Represents a Stochastic Gradient Descent (SGD) optimizer for machine learning models.

## For Beginners

Think of this optimizer as a hiker trying to find the lowest point in a hilly landscape: - The hiker (optimizer) takes steps downhill to find the lowest point (best model parameters) - Instead of looking at the entire landscape at once, the hiker looks at small patches (subsets of data) - The hiker adjusts their step size (learning rate) as they go - This approach helps the hiker find a good low point quickly, even in a complex landscape This method is efficient for large datasets and can often find good solutions quickly.

## How It Works

The StochasticGradientDescentOptimizer is a gradient-based optimization algorithm that iteratively adjusts model parameters to minimize the loss function. It uses a stochastic approach, updating parameters based on a subset of the training data in each iteration.

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
    .ConfigureOptimizer(new StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with StochasticGradientDescentOptimizer.");
```

