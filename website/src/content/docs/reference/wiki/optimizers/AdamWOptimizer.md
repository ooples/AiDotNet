---
title: "AdamWOptimizer<T, TInput, TOutput>"
description: "Implements the AdamW (Adam with decoupled Weight decay) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the AdamW (Adam with decoupled Weight decay) optimization algorithm.

## For Beginners

AdamW is like Adam but handles regularization (preventing overfitting) in a smarter way.
The difference might seem technical, but AdamW consistently achieves better results on tasks like training transformers
and large neural networks. If you're choosing between Adam and AdamW, AdamW is generally the better choice.

## How It Works

AdamW is a variant of Adam that fixes the weight decay implementation. In standard Adam with L2 regularization,
weight decay is coupled with the adaptive learning rate, which can lead to suboptimal regularization effects.
AdamW decouples weight decay from the gradient-based update, applying it directly to the weights.

The key difference:

- Adam with L2: gradient = gradient + lambda * weights (then apply Adam update)
- AdamW: weights = weights - lr * adam_update - lr * lambda * weights (decoupled)

Based on the paper "Decoupled Weight Decay Regularization" by Ilya Loshchilov and Frank Hutter.

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdamWOptimizer(IFullModel<,,>,AdamWOptimizerOptions<,,>)` | Initializes a new instance of the AdamWOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |
| `UseAMSGrad` | Gets whether AMSGrad variant is enabled. |
| `WeightDecay` | Gets the current weight decay coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` |  |
| `Deserialize(Byte[])` | Deserializes the optimizer's state from a byte array. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetOptions` | Gets the current optimizer options. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the AdamW optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes AdamW optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the AdamW algorithm. |
| `Reset` | Resets the optimizer's internal state. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses an AdamW gradient update to recover original parameters. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates a matrix of parameters using the AdamW optimization algorithm. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the AdamW optimization algorithm with decoupled weight decay. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using the AdamW kernel. |
| `UpdateParametersInternal(Vector<>,Vector<>)` | Internal method to update parameters without reinitializing moment vectors. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the AdamW update rule with decoupled weight decay. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentBeta1` | The current value of beta1 (exponential decay rate for first moment estimates). |
| `_currentBeta2` | The current value of beta2 (exponential decay rate for second moment estimates). |
| `_gpuM` | GPU buffer for first moment estimates (m). |
| `_gpuV` | GPU buffer for second moment estimates (v). |
| `_m` | The first moment vector (moving average of gradients). |
| `_options` | The options specific to the AdamW optimizer. |
| `_previousM` | Stores the pre-update snapshot of first moment vector for accurate reverse updates. |
| `_previousT` | Stores the pre-update timestep for accurate reverse updates. |
| `_previousV` | Stores the pre-update snapshot of second moment vector for accurate reverse updates. |
| `_t` | The current time step (iteration count). |
| `_v` | The second moment vector (moving average of squared gradients). |
| `_vMax` | Maximum of past squared gradients (used when AMSGrad is enabled). |

