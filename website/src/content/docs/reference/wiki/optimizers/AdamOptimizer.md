---
title: "AdamOptimizer<T, TInput, TOutput>"
description: "Implements the Adam (Adaptive Moment Estimation) optimization algorithm for gradient-based optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Adam (Adaptive Moment Estimation) optimization algorithm for gradient-based optimization.

## For Beginners

Adam is like a smart personal trainer for your machine learning model.
It helps your model learn efficiently by adjusting how it learns based on past experiences.

## How It Works

Adam is an advanced optimization algorithm that combines ideas from RMSprop and Momentum optimization methods.
It adapts the learning rates for each parameter individually and is well-suited for problems with noisy or sparse gradients.

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
    .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdamOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdamOptimizer(IFullModel<,,>,AdamOptimizerOptions<,,>)` | Initializes a new instance of the AdamOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` |  |
| `AnyGradientIsAnomalous(TapeStepContext<>)` | Scans every gradient tensor for NaN/Inf entries and returns true on the first sighting. |
| `AnyGradientIsAnomalous(Vector<>)` | Flat-vector overload of `TapeStepContext{` for the Optimize / UpdateSolution path (#1380 part 2). |
| `Deserialize(Byte[])` | Deserializes the optimizer's state from a byte array. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetOptions` | Gets the current optimizer options. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the Adam optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes Adam optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Adam algorithm. |
| `Reset` | Resets the optimizer's internal state. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `ShouldRunAnomalyGuard` | Applies PyTorch-style global-norm gradient clipping across every gradient in the tape step's `Gradients` dictionary. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates a matrix of parameters using the Adam optimization algorithm. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the Adam optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using the Adam kernel. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentBeta1` | The current value of beta1 (exponential decay rate for first moment estimates). |
| `_currentBeta2` | The current value of beta2 (exponential decay rate for second moment estimates). |
| `_gpuM` | GPU buffer for first moment estimates (m). |
| `_gpuV` | GPU buffer for second moment estimates (v). |
| `_m` | The first moment vector (moving average of gradients). |
| `_options` | The options specific to the Adam optimizer. |
| `_previousM` | Stores the pre-update snapshot of first moment vector for accurate reverse updates. |
| `_previousT` | Stores the pre-update timestep for accurate reverse updates. |
| `_previousV` | Stores the pre-update snapshot of second moment vector for accurate reverse updates. |
| `_t` | The current time step (iteration count). |
| `_tapeVMax` | Per-parameter running maximum of v̂_t when AMSGrad is enabled (Reddi, Kale, Kumar 2018). |
| `_v` | The second moment vector (moving average of squared gradients). |
| `_vMaxVector` | Running maximum of v̂ when AMSGrad is enabled, for the Vector-based UpdateParameters / UpdateSolution paths (the tape-based Step path tracks vMax per-tensor in `_tapeVMax`). |

