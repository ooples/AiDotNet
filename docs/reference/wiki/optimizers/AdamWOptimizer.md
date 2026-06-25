---
title: "AdamWOptimizer"
description: "Implements the AdamW (Adam with decoupled Weight decay) optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the AdamW (Adam with decoupled Weight decay) optimization algorithm.

## For Beginners

AdamW is like Adam but handles regularization (preventing overfitting) in a smarter way. The difference might seem technical, but AdamW consistently achieves better results on tasks like training transformers and large neural networks. If you're choosing between Adam and AdamW, AdamW is generally the better choice.

## How It Works

AdamW is a variant of Adam that fixes the weight decay implementation. In standard Adam with L2 regularization, weight decay is coupled with the adaptive learning rate, which can lead to suboptimal regularization effects. AdamW decouples weight decay from the gradient-based update, applying it directly to the weights. 

The key difference: - Adam with L2: gradient = gradient + lambda * weights (then apply Adam update) - AdamW: weights = weights - lr * adam_update - lr * lambda * weights (decoupled)

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
    .ConfigureOptimizer(new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdamWOptimizer.");
```

