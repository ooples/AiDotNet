---
title: "Adam8BitOptimizer"
description: "Implements an 8-bit quantized Adam optimizer that reduces memory usage by storing optimizer states in 8-bit format."
section: "Reference"
---

_Optimizers_

Implements an 8-bit quantized Adam optimizer that reduces memory usage by storing optimizer states in 8-bit format.

## For Beginners

When training a neural network, the optimizer needs to remember information about
past gradients. Standard Adam stores two numbers per parameter (momentum and variance), which can use a lot of
memory for large models. 8-bit Adam compresses these numbers, similar to how images are compressed, reducing
memory usage while maintaining training quality.

## How It Works

8-bit Adam provides the same optimization algorithm as standard Adam but uses quantized 8-bit representations
for storing the first moment (m) and second moment (v) estimates. This reduces memory usage by approximately
4x for optimizer states, which is particularly beneficial when training large models.

**How It Works:**

- Optimizer states are divided into blocks (default 2048 elements each)
- Each block has its own scaling factor for accurate quantization
- States are dequantized before computing updates, then requantized after
- The actual parameter updates use full precision for accuracy

**When to Use:**

- Training large models where optimizer memory is a bottleneck
- GPU training with limited VRAM
- Distributed training where memory per GPU is constrained

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
    .ConfigureOptimizer(new Adam8BitOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with Adam8BitOptimizer.");
```

