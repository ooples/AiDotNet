---
title: "AMSGradOptimizer<T, TInput, TOutput>"
description: "Implements the AMSGrad optimization algorithm, an improved version of Adam optimizer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the AMSGrad optimization algorithm, an improved version of Adam optimizer.

## For Beginners

AMSGrad is like a smart assistant that helps adjust the learning process.
It remembers past information to make better decisions about how quickly to learn in different parts of the problem.

## How It Works

AMSGrad is an adaptive learning rate optimization algorithm that addresses some of the convergence issues in Adam.
It maintains the maximum of past squared gradients to ensure non-decreasing step sizes.

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
    .ConfigureOptimizer(new AMSGradOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AMSGradOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AMSGradOptimizer(IFullModel<,,>,AMSGradOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the AMSGradOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` | Describes this AMSGrad optimizer for the compiled fused-training kernel. |
| `Deserialize(Byte[])` | Restores the optimizer's state from a byte array previously created by the Serialize method. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients based on the current state of the optimizer and input data. |
| `GetOptions` | Retrieves the current options of the optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the AMSGrad optimizer. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the AMSGrad algorithm. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses an AMSGrad gradient update to recover original parameters. |
| `Serialize` | Converts the current state of the optimizer into a byte array for storage or transmission. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the AMSGrad optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | GPU-accelerated parameter update for AMSGrad optimizer. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the AMSGrad update rule. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_m` | The first moment vector (moving average of gradients). |
| `_options` | The options specific to the AMSGrad optimizer. |
| `_t` | The current time step. |
| `_v` | The second moment vector (moving average of squared gradients). |
| `_vHat` | The maximum of past second moments. |

