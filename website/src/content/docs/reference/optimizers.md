---
title: "Optimizers"
description: "Optimizer reference for training neural networks."
order: 3
section: "Reference"
---


Reference for the optimization algorithms and learning-rate schedulers in AiDotNet.

Optimizers are generic over `<T, TInput, TOutput>` and take the model they optimize: `new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(model)`. Tuning (learning rate, betas, weight decay) lives in the matching `*Options`. Plug an optimizer in with `ConfigureOptimizer(...)`.

---

## First-Order Optimizers

| Optimizer | Use Case |
|:----------|:---------|
| `StochasticGradientDescentOptimizer<T,TIn,TOut>` | General purpose |
| `MomentumOptimizer<T,TIn,TOut>` | Faster convergence |
| `NesterovAcceleratedGradientOptimizer<T,TIn,TOut>` | Look-ahead momentum |
| `AdamOptimizer<T,TIn,TOut>` | General deep learning |
| `AdamWOptimizer<T,TIn,TOut>` | Transformers, large models |
| `AdaMaxOptimizer<T,TIn,TOut>` | Sparse gradients |
| `NadamOptimizer<T,TIn,TOut>` | Improved convergence |
| `AMSGradOptimizer<T,TIn,TOut>` | Stable training |
| `AdagradOptimizer<T,TIn,TOut>` | Sparse features |
| `AdaDeltaOptimizer<T,TIn,TOut>` | No learning-rate tuning |
| `RootMeanSquarePropagationOptimizer<T,TIn,TOut>` | RNNs (RMSProp) |
| `FTRLOptimizer<T,TIn,TOut>` | Online learning |
| `LARSOptimizer<T,TIn,TOut>` | Large-batch training |
| `LAMBOptimizer<T,TIn,TOut>` | BERT pre-training |
| `LionOptimizer<T,TIn,TOut>` | Vision/language models |

---

## Second-Order Optimizers

| Optimizer | Use Case |
|:----------|:---------|
| `LBFGSOptimizer<T,TIn,TOut>` | Small models, full batch |
| `BFGSOptimizer<T,TIn,TOut>` | Convex optimization |
| `NewtonMethodOptimizer<T,TIn,TOut>` | Quadratic convergence |
| `LevenbergMarquardtOptimizer<T,TIn,TOut>` | Nonlinear least squares |
| `ConjugateGradientOptimizer<T,TIn,TOut>` | Large sparse systems |
| `TrustRegionOptimizer<T,TIn,TOut>` | Robust convergence |

---

## Evolutionary Optimizers

| Optimizer | Use Case |
|:----------|:---------|
| `GeneticAlgorithmOptimizer<T,TIn,TOut>` | Complex search spaces |
| `CMAESOptimizer<T,TIn,TOut>` | Black-box optimization |
| `ParticleSwarmOptimizer<T,TIn,TOut>` | Global optimization |
| `DifferentialEvolutionOptimizer<T,TIn,TOut>` | Continuous optimization |
| `SimulatedAnnealingOptimizer<T,TIn,TOut>` | Combinatorial optimization |

---

## Learning-Rate Schedulers

Schedulers implement `ILearningRateScheduler` and attach via `ConfigureLearningRateScheduler(...)`.

| Scheduler | Description |
|:----------|:------------|
| `StepLRScheduler` | Decay by factor at fixed intervals |
| `MultiStepLRScheduler` | Decay at specified milestones |
| `ExponentialLRScheduler` | Exponential decay |
| `CosineAnnealingLRScheduler` | Cosine annealing |
| `LinearLRScheduler` | Linear decay |
| `ReduceLROnPlateauScheduler` | Reduce when a metric plateaus |
| `OneCycleLRScheduler` | One-cycle policy |

---

## Using an Optimizer

Configure the model first, then hand the same model to the optimizer.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(0);
var trainX = new Tensor<float>(new[] { 64, 16 });
var trainY = new Tensor<float>(new[] { 64, 3 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 16; j++) trainX[new[] { i, j }] = (float)rng.NextDouble();
    trainY[new[] { i, i % 3 }] = 1f;
}

var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: 16, numClasses: 3, complexity: NetworkComplexity.Simple));

// Swap AdamWOptimizer for any optimizer above — they share the (model, options) shape.
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Trained; output [{string.Join(", ", result.Predict(trainX).Shape)}]");
```

### Tuning via Options

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

var trainX = new Tensor<float>(new[] { 32, 8 });
var trainY = new Tensor<float>(new[] { 32, 2 });
for (int i = 0; i < 32; i++) { trainX[new[] { i, 0 }] = i / 32f; trainY[new[] { i, i % 2 }] = 1f; }

var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: 8, numClasses: 2, complexity: NetworkComplexity.Simple));

var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>();

var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamOptimizer<float, Tensor<float>, Tensor<float>>(model, options))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with custom optimizer options.");
```

---

## Optimizer Selection Guide

| Task | Recommended |
|:-----|:------------|
| General deep learning | `AdamWOptimizer` |
| Transformers / LLMs | `AdamWOptimizer` + cosine schedule |
| Large-batch training | `LAMBOptimizer` |
| Vision models | `StochasticGradientDescentOptimizer` (momentum) or `LionOptimizer` |
| RNNs | `RootMeanSquarePropagationOptimizer` or `AdamOptimizer` |
| Small datasets | `LBFGSOptimizer` |
| Hyperparameter search | `GeneticAlgorithmOptimizer` / `ParticleSwarmOptimizer` |

## Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|:----------|:--------------|:------|
| Learning rate | 1e-5 to 1e-2 | Start with 3e-4 for Adam |
| Weight decay | 0 to 0.1 | 0.01 is common |
| Beta1 | 0.9 to 0.95 | 0.9 is standard |
| Beta2 | 0.99 to 0.999 | 0.999 for Adam |
| Epsilon | 1e-8 to 1e-6 | 1e-8 is standard |
