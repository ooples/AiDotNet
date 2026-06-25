---
title: "LAMBOptimizer"
description: "Implements the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm.

## For Beginners

LAMB is the optimizer of choice for training large language models like BERT with massive batch sizes. It works by: 
- Computing Adam-style updates (momentum + adaptive learning rates)
- Adding weight decay to prevent overfitting
- Scaling the update per-layer based on weight/update magnitude ratios
This combination allows training to scale linearly with batch size while maintaining the same final accuracy as small-batch training.

## How It Works

LAMB combines Adam's adaptive learning rates with LARS's layer-wise scaling, enabling training with extremely large batch sizes (up to 32K) while maintaining accuracy. 

**Key Formula:**

m = beta1 * m + (1 - beta1) * g v = beta2 * v + (1 - beta2) * g^2 m_hat = m / (1 - beta1^t) v_hat = v / (1 - beta2^t) r = m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w trust_ratio = ||w|| / ||r|| w = w - lr * trust_ratio * r

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
    .ConfigureOptimizer(new LAMBOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LAMBOptimizer.");
```

