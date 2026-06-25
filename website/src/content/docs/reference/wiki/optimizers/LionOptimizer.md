---
title: "LionOptimizer<T, TInput, TOutput>"
description: "Implements the Lion (Evolved Sign Momentum) optimization algorithm for gradient-based optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Lion (Evolved Sign Momentum) optimization algorithm for gradient-based optimization.

## For Beginners

Lion is like a simplified but more powerful version of Adam. Instead of
carefully measuring how big each step should be (like Adam does), Lion only looks at which direction
to go and takes consistent-sized steps in that direction. This is like following a compass that only
shows direction - it's simpler, uses less memory, and often gets you to your destination faster.
Lion is particularly good for training large neural networks.

## How It Works

Lion is a modern optimization algorithm discovered through symbolic program search that offers significant
advantages over traditional optimizers like Adam. It achieves 50% memory reduction by maintaining only a
single momentum state (compared to Adam's two states) while often achieving superior performance on large
transformer models and other deep learning architectures.

The algorithm uses sign-based gradient updates, which provides implicit regularization and better
generalization. Unlike Adam's magnitude-based updates, Lion focuses purely on the direction of gradients,
making it more robust to gradient scale variations and leading to more consistent training dynamics.

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
    .ConfigureOptimizer(new LionOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LionOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LionOptimizer(IFullModel<,,>,LionOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the LionOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` | Describes this Lion instance for the fused kernel (Tensors `OptimizerType.Lion` = `LionUpdateSimd(lr, b1, b2, wd)`): Beta1/Beta2 → β1/β2, WeightDecay → wd. |
| `Deserialize(Byte[])` | Deserializes the optimizer's state from a byte array. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetOptions` | Gets the current optimizer options. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the Lion optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes Lion optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Lion algorithm. |
| `Reset` | Resets the optimizer's internal state. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a Lion gradient update to recover original parameters. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates a matrix of parameters using the Lion optimization algorithm. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the Lion optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on GPU using Lion optimization. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the Lion update rule. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentBeta1` | The current value of beta1 (interpolation momentum). |
| `_currentBeta2` | The current value of beta2 (update momentum). |
| `_gpuM` | GPU buffer for momentum estimates. |
| `_m` | The momentum vector (exponentially moving average of gradients). |
| `_options` | The options specific to the Lion optimizer. |
| `_t` | The current time step (iteration count). |

