---
title: "LARSOptimizer"
description: "Implements the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm.

## For Beginners

When training with very large batches (common in self-supervised
learning like SimCLR), regular optimizers can become unstable because gradients get averaged
over more samples, making them smaller. LARS solves this by looking at each layer and asking
"how big are the weights compared to the gradients?" and scaling the learning rate accordingly.
This allows stable training with batch sizes of 4096 or even larger.

## How It Works

LARS is specifically designed for training with very large batch sizes (4096-32768).
It automatically adapts the learning rate for each layer based on the ratio of
parameter norm to gradient norm, which helps maintain stable training at scale.

**Key Formula:**

local_lr = trust_coeff * ||w|| / (||g|| + weight_decay * ||w|| + epsilon)
update = local_lr * (g + weight_decay * w)
w = w - lr * update (with momentum)

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
    .ConfigureOptimizer(new LARSOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LARSOptimizer.");
```

