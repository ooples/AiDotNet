---
title: "LAMBOptimizer<T, TInput, TOutput>"
description: "Implements the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm.

## For Beginners

LAMB is the optimizer of choice for training large language models
like BERT with massive batch sizes. It works by:

- Computing Adam-style updates (momentum + adaptive learning rates)
- Adding weight decay to prevent overfitting
- Scaling the update per-layer based on weight/update magnitude ratios

This combination allows training to scale linearly with batch size while maintaining
the same final accuracy as small-batch training.

## How It Works

LAMB combines Adam's adaptive learning rates with LARS's layer-wise scaling, enabling
training with extremely large batch sizes (up to 32K) while maintaining accuracy.

**Key Formula:**

Based on the paper "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
by You et al. (2019).

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LAMBOptimizer(IFullModel<,,>,LAMBOptimizerOptions<,,>)` | Initializes a new instance of the LAMBOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta1` | Gets the current beta1 value. |
| `Beta2` | Gets the current beta2 value. |
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |
| `WeightDecay` | Gets the current weight decay coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` | Describes this LAMB instance for the fused kernel (Tensors `OptimizerType.LAMB` = `LAMBUpdateSimd(lr, b1, b2, eps, wd)`): Beta1/Beta2 → β1/β2, Epsilon → eps, WeightDecay → wd. |
| `Deserialize(Byte[])` | Deserializes the optimizer's state from a byte array. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `ExtractLayerVector(Vector<>,Int32,Int32)` | Extracts a layer's parameters from the full parameter vector. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetDataSize(OptimizationInputData<,,>)` | Gets the size of the training data. |
| `GetOptions` | Gets the current optimizer options. |
| `GetWarmupLearningRate` | Gets the learning rate with warmup applied. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the LAMB optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes LAMB optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the LAMB algorithm. |
| `Reset` | Resets the optimizer's internal state. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a LAMB gradient update to recover original parameters. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateLayerVector(Vector<>,Vector<>,Int32,Int32)` | Updates a layer's values in the full parameter vector. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates a matrix of parameters using the LAMB optimization algorithm. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the LAMB optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using the LAMB kernel. |
| `UpdateParametersLAMB(Vector<>,Vector<>,Double)` | Internal LAMB parameter update. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the LAMB update rule. |
| `UpdateSolutionWithLAMB(IFullModel<,,>,Vector<>,Double)` | Updates the solution using the LAMB algorithm. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuM` | GPU buffer for first moment estimates (m). |
| `_gpuV` | GPU buffer for second moment estimates (v). |
| `_m` | The first moment vector (moving average of gradients). |
| `_options` | The options specific to the LAMB optimizer. |
| `_previousM` | Previous first moment for reverse updates. |
| `_previousT` | Previous step count for reverse updates. |
| `_previousV` | Previous second moment for reverse updates. |
| `_t` | The current time step (iteration count). |
| `_v` | The second moment vector (moving average of squared gradients). |
| `_warmupSteps` | The total number of warmup steps. |

