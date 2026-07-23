---
title: "Mixture of Experts"
description: "Train a Mixture-of-Experts network through the facade."
order: 5
section: "Examples"
---


This guide shows how to train a Mixture-of-Experts (MoE) network with AiDotNet. An MoE routes each input to a few specialized "expert" sub-networks via a learned gate, giving large model capacity at a fraction of the per-input compute.

## Overview

`MixtureOfExpertsNeuralNetwork<T>` is a regular AiDotNet neural network, so it builds the same way as every other model: `ConfigureModel(...)` + `ConfigureDataLoader(...)` + `BuildAsync()`, then `result.Predict(...)`.

## Standard Pattern

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// 200 samples of 16 features, one-hot labels for 4 classes.
var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 200, 16 });
var trainY = new Tensor<double>(new[] { 200, 4 });
for (int i = 0; i < 200; i++)
{
    for (int j = 0; j < 16; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}

// 4 experts, route each input to its top 2.
var moe = new MixtureOfExpertsNeuralNetwork<double>(
    new MixtureOfExpertsOptions<double> { NumExperts = 4, TopK = 2, InputDim = 16, OutputDim = 4 },
    new NeuralNetworkArchitecture<double>(inputFeatures: 16, numClasses: 4, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(moe)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

var scores = result.Predict(trainX);
Console.WriteLine($"Output shape: [{string.Join(", ", scores.Shape)}]");
```

## Complete Example: Classification

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(7);
var trainX = new Tensor<double>(new[] { 300, 20 });
var trainY = new Tensor<double>(new[] { 300, 3 });
for (int i = 0; i < 300; i++)
{
    for (int j = 0; j < 20; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 3 }] = 1.0;
}

var moe = new MixtureOfExpertsNeuralNetwork<double>(
    new MixtureOfExpertsOptions<double> { NumExperts = 6, TopK = 2, InputDim = 20, OutputDim = 3 },
    new NeuralNetworkArchitecture<double>(inputFeatures: 20, numClasses: 3, complexity: NetworkComplexity.Medium));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(moe)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

// Predict class scores for a batch and read the argmax of row 0.
var scores = result.Predict(trainX);
int predicted = 0;
for (int c = 1; c < 3; c++)
    if (scores[new[] { 0, c }] > scores[new[] { 0, predicted }]) predicted = c;

Console.WriteLine($"Predicted class for sample 0: {predicted}");
Console.WriteLine($"Layers: {result.LayerCount}, params: {result.TotalTrainableParameters:N0}");
```

## Configuration Options

`MixtureOfExpertsOptions` controls the experts and the gate.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var options = new MixtureOfExpertsOptions<double>
{
    NumExperts = 8,            // total experts
    TopK = 2,                  // experts each input is routed to
    InputDim = 32,             // expert input width
    OutputDim = 4,             // expert output width
    HiddenExpansion = 4,       // hidden size = InputDim * HiddenExpansion
    UseLoadBalancing = true,   // encourage even expert utilization
    LoadBalancingWeight = 0.01 // strength of the load-balancing penalty
};

var rng = new Random(1);
var trainX = new Tensor<double>(new[] { 128, 32 });
var trainY = new Tensor<double>(new[] { 128, 4 });
for (int i = 0; i < 128; i++)
{
    for (int j = 0; j < 32; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(new MixtureOfExpertsNeuralNetwork<double>(
        options,
        new NeuralNetworkArchitecture<double>(inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple)))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Trained MoE with {options.NumExperts} experts (top-{options.TopK} routing).");
```

## Regression Example

For a regression MoE, build the architecture with a regression task and a single output.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(3);
var trainX = new Tensor<double>(new[] { 150, 8 });
var trainY = new Tensor<double>(new[] { 150, 1 });
for (int i = 0; i < 150; i++)
{
    double sum = 0;
    for (int j = 0; j < 8; j++) { double v = rng.NextDouble(); trainX[new[] { i, j }] = v; sum += v; }
    trainY[new[] { i, 0 }] = sum / 8.0;
}

var moe = new MixtureOfExpertsNeuralNetwork<double>(
    new MixtureOfExpertsOptions<double> { NumExperts = 4, TopK = 2, InputDim = 8, OutputDim = 1 },
    new NeuralNetworkArchitecture<double>(
        inputType: InputType.OneDimensional,
        taskType: NeuralNetworkTaskType.Regression,
        inputSize: 8,
        outputSize: 1));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(moe)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

var prediction = result.Predict(trainX);
Console.WriteLine($"Predicted value for sample 0: {prediction[new[] { 0, 0 }]:F4}");
```

## Comparison with a Plain Network

Because an MoE is just another model behind the facade, swapping it for a dense network is a one-line change — train both and compare.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(11);
var trainX = new Tensor<double>(new[] { 200, 16 });
var trainY = new Tensor<double>(new[] { 200, 4 });
for (int i = 0; i < 200; i++)
{
    for (int j = 0; j < 16; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}
var loader = DataLoaders.FromTensors(trainX, trainY);
var arch = new NeuralNetworkArchitecture<double>(inputFeatures: 16, numClasses: 4, complexity: NetworkComplexity.Simple);

// Mixture of Experts
var moeResult = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(new MixtureOfExpertsNeuralNetwork<double>(
        new MixtureOfExpertsOptions<double> { NumExperts = 4, TopK = 2, InputDim = 16, OutputDim = 4 }, arch))
    .ConfigureDataLoader(loader)
    .BuildAsync();

// Plain dense network — same facade call, different ConfigureModel argument.
var denseResult = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(new NeuralNetwork<double>(
        new NeuralNetworkArchitecture<double>(inputFeatures: 16, numClasses: 4, complexity: NetworkComplexity.Simple)))
    .ConfigureDataLoader(loader)
    .BuildAsync();

Console.WriteLine($"MoE params:   {moeResult.TotalTrainableParameters:N0}");
Console.WriteLine($"Dense params: {denseResult.TotalTrainableParameters:N0}");
```

## Best Practices

1. **Start with a few experts**: 4–8 experts with top-2 routing is a strong baseline.
2. **Keep load balancing on**: `UseLoadBalancing` stops a handful of experts from dominating.
3. **Match dims to your data**: set `InputDim` / `OutputDim` to your feature and class counts.
4. **Scale experts, not depth**: MoE adds capacity by adding experts, keeping per-input compute low.
5. **Compare against a dense baseline**: it is one line to swap models behind the facade.

## Summary

`MixtureOfExpertsNeuralNetwork<T>` brings sparse expert routing to AiDotNet through the same `AiModelBuilder` flow as any other model:

- `ConfigureModel(new MixtureOfExpertsNeuralNetwork<T>(options, architecture))`
- Tune experts and the gate with `MixtureOfExpertsOptions` (`NumExperts`, `TopK`, load balancing)
- Train and predict exactly like a dense network — `BuildAsync()` then `result.Predict(...)`
