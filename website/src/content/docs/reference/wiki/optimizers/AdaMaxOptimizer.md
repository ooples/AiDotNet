---
title: "AdaMaxOptimizer<T, TInput, TOutput>"
description: "Represents an AdaMax optimizer, an extension of Adam that uses the infinity norm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Represents an AdaMax optimizer, an extension of Adam that uses the infinity norm.

## For Beginners

AdaMax is like a smart learning assistant that adjusts its learning speed
for each piece of information it's trying to learn. It's particularly good at handling different
scales of information without getting confused.

Key features:

- Adapts the learning rate for each parameter
- Uses the maximum (infinity norm) of past gradients, which can be more stable
- Good for problems where the gradients can be sparse or have different scales

## How It Works

AdaMax is an adaptive learning rate optimization algorithm that extends the Adam optimizer.
It uses the infinity norm to update parameters, which can make it more robust in certain scenarios.

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
    .ConfigureOptimizer(new AdaMaxOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdaMaxOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaMaxOptimizer(IFullModel<,,>,AdaMaxOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the AdaMaxOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` | Describes this AdaMax instance for the fused-compiled training kernel (Tensors `OptimizerType.AdaMax`). |
| `Deserialize(Byte[])` | Restores the optimizer's state from a byte array created by the Serialize method. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients specific to the AdaMax optimizer. |
| `GetOptions` | Gets the current options of the AdaMax optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters for the AdaMax optimizer. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the AdaMax algorithm. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses an AdaMax gradient update to recover original parameters. |
| `Serialize` | Converts the current state of the optimizer into a byte array for storage or transmission. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer options with new AdaMax-specific options. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the AdaMax optimization algorithm. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the AdaMax update rule. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_m` | The first moment vector that tracks the exponentially weighted moving average of gradients. |
| `_options` | The configuration options specific to the AdaMax optimizer. |
| `_t` | The current time step or iteration counter. |
| `_u` | The exponentially weighted infinity norm of past gradients. |

