# Mixture-of-Experts (MoE) Usage Guide

This guide demonstrates how to use Mixture-of-Experts layers with AiDotNet's PredictionModelBuilder.

## Overview

Mixture-of-Experts (MoE) is a powerful architecture that enables models with extremely high capacity while remaining computationally efficient by activating only a subset of parameters per input.

## Standard Pattern (Same as All Neural Networks)

MoE follows the exact same pattern as other neural network models in AiDotNet:

```csharp
using AiDotNet;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Models;

// 1. Create MoE layer using the builder
var moeLayer = new MixtureOfExpertsBuilder<float>()
    .WithExperts(8)                 // 8 specialist networks
    .WithDimensions(128, 128)       // Input/output dimensions
    .WithTopK(2)                    // Use top 2 experts per input
    .WithLoadBalancing(true, 0.01)  // Enable load balancing
    .Build();

// 2. Create output layer
var outputLayer = new DenseLayer<float>(128, 10, new SoftmaxActivation<float>());

// 3. Create architecture with your layers
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.Medium,
    inputSize: 128,
    outputSize: 10,
    layers: new List<ILayer<float>> { moeLayer, outputLayer }
);

// 4. Wrap in NeuralNetworkModel (same as always)
var model = new NeuralNetworkModel<float>(architecture);

// 5. Use with PredictionModelBuilder (same as always)
var builder = new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = builder
    .ConfigureModel(model)
    .Build(trainingData, trainingLabels);

// 6. Make predictions (same as always)
var predictions = builder.Predict(testData, result);
```

## Complete Example: Classification Task

```csharp
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

// Prepare your data
int numSamples = 1000;
int numFeatures = 784;  // e.g., 28x28 images flattened
int numClasses = 10;

var trainingData = new Tensor<float>(new[] { numSamples, numFeatures });
var trainingLabels = new Tensor<float>(new[] { numSamples, numClasses });
// ... fill with actual data ...

// Create MoE layer
var moeLayer = new MixtureOfExpertsBuilder<float>()
    .WithExperts(8)
    .WithDimensions(numFeatures, numFeatures)
    .WithTopK(2)
    .WithLoadBalancing(true)
    .Build();

// Create output layer
var outputLayer = new DenseLayer<float>(
    numFeatures,
    numClasses,
    new SoftmaxActivation<float>()
);

// Assemble architecture
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.Medium,
    inputSize: numFeatures,
    outputSize: numClasses,
    layers: new List<ILayer<float>> { moeLayer, outputLayer }
);

// Create model
var model = new NeuralNetworkModel<float>(architecture);

// Train with PredictionModelBuilder
var builder = new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = builder
    .ConfigureModel(model)
    .Build(trainingData, trainingLabels);

// Evaluate
Console.WriteLine($"Training Accuracy: {result.TrainingAccuracy:P2}");
Console.WriteLine($"Validation Accuracy: {result.ValidationAccuracy:P2}");

// Make predictions
var testData = new Tensor<float>(new[] { 100, numFeatures });
var predictions = builder.Predict(testData, result);

// Save model
builder.SaveModel(result, "moe_model.bin");

// Load and use later
var loadedModel = builder.LoadModel("moe_model.bin");
var newPredictions = builder.Predict(testData, loadedModel);
```

## Deep MoE Architecture (Multiple Layers)

Stack multiple MoE layers for deeper architectures:

```csharp
var layers = new List<ILayer<float>>();

// Input projection
layers.Add(new DenseLayer<float>(784, 256, new ReLUActivation<float>()));

// Stack multiple MoE layers
for (int i = 0; i < 3; i++)
{
    var moeLayer = new MixtureOfExpertsBuilder<float>()
        .WithExperts(8)
        .WithDimensions(256, 256)
        .WithTopK(2)
        .WithLoadBalancing(true)
        .Build();

    layers.Add(moeLayer);
}

// Output layer
layers.Add(new DenseLayer<float>(256, 10, new SoftmaxActivation<float>()));

// Create architecture
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.Complex,
    inputSize: 784,
    outputSize: 10,
    layers: layers
);

var model = new NeuralNetworkModel<float>(architecture);
var result = builder.ConfigureModel(model).Build(trainingData, trainingLabels);
```

## Configuring the MoE Builder

The `MixtureOfExpertsBuilder` provides a fluent API for creating MoE layers:

```csharp
var moeLayer = new MixtureOfExpertsBuilder<float>()
    .WithExperts(8)                    // Number of expert networks
    .WithDimensions(128, 128)          // Input and output dimensions
    .WithExpertHiddenDim(512)          // Hidden size in experts (4x expansion)
    .WithTopK(2)                       // Sparse routing: top 2 experts
    .WithLoadBalancing(true, 0.01)     // Enable load balancing
    .WithExpertActivation(new GELUActivation<float>())  // Custom activation
    .WithOutputActivation(new IdentityActivation<float>())
    .WithIntermediateLayer(true)       // Use 2-layer experts
    .Build();
```

### Parameter Guidelines

**Number of Experts:**
- 2-4: Small models, limited compute
- 4-8: Most applications (recommended)
- 8-16: Larger, more complex tasks
- 16+: Very large models (use with TopK)

**Top-K Selection:**
- `topK = 0`: All experts active (soft routing) - best for 4-8 experts
- `topK = 1`: Only best expert - very fast, for 32+ experts
- `topK = 2`: Top 2 experts - good balance for 8-32 experts

**Load Balancing Weight:**
- `0.01`: Gentle (default, works well)
- `0.05`: Moderate (if you see imbalance)
- `0.1`: Strong (may reduce accuracy slightly)

## Monitoring Expert Usage

Access diagnostics to monitor expert balance during training:

```csharp
using AiDotNet.Interfaces;

// After training, get the underlying network
var network = ((NeuralNetworkModel<float>)result.Model).Network;

// Find MoE layers and check diagnostics
foreach (var layer in network.Layers)
{
    if (layer is IAuxiliaryLossLayer<float> auxLayer)
    {
        var diagnostics = auxLayer.GetAuxiliaryLossDiagnostics();

        Console.WriteLine("\nExpert Usage Statistics:");
        foreach (var (key, value) in diagnostics)
        {
            Console.WriteLine($"  {key}: {value}");
        }
    }
}
```

## Advanced: Custom Expert Architectures

Create custom expert networks instead of using the builder:

```csharp
// Create custom experts
var experts = new List<ILayer<float>>();
for (int i = 0; i < 8; i++)
{
    var expertLayers = new List<ILayer<float>>
    {
        new DenseLayer<float>(128, 512, new ReLUActivation<float>()),
        new DenseLayer<float>(512, 256, new ReLUActivation<float>()),
        new DenseLayer<float>(256, 128, new IdentityActivation<float>())
    };

    experts.Add(new Expert<float>(expertLayers, new[] { 128 }, new[] { 128 }));
}

// Create router
var router = new DenseLayer<float>(128, 8); // 8 outputs for 8 experts

// Create MoE layer directly
var moeLayer = new MixtureOfExpertsLayer<float>(
    experts,
    router,
    inputShape: new[] { 128 },
    outputShape: new[] { 128 },
    topK: 2,
    activationFunction: new IdentityActivation<float>(),
    useLoadBalancing: true,
    loadBalancingWeight: 0.01f
);

// Use in architecture as before
var layers = new List<ILayer<float>> { moeLayer, outputLayer };
var architecture = new NeuralNetworkArchitecture<float>(...);
```

## Regression Example

MoE works for regression too - just change the task type and output activation:

```csharp
var moeLayer = new MixtureOfExpertsBuilder<float>()
    .WithExperts(4)
    .WithDimensions(10, 10)
    .Build();

var outputLayer = new DenseLayer<float>(10, 1, new IdentityActivation<float>());

var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.Regression,  // Regression task
    inputSize: 10,
    outputSize: 1,
    layers: new List<ILayer<float>> { moeLayer, outputLayer }
);

var model = new NeuralNetworkModel<float>(architecture);
var result = builder.ConfigureModel(model).Build(trainingData, trainingTargets);
```

## Key Points

1. **Same Pattern**: MoE uses the exact same pattern as all neural networks in AiDotNet
2. **Layer Integration**: MoE layers work like any other layer (Dense, Conv, etc.)
3. **Builder Pattern**: Use MixtureOfExpertsBuilder for convenient MoE layer creation
4. **Architecture Flexibility**: Combine MoE with any other layers
5. **Automatic Training**: PredictionModelBuilder handles all training automatically

## Common Patterns

### Simple MoE Network
```csharp
layers: { MoELayer, OutputLayer }
```

### Deep MoE Network
```csharp
layers: { InputProjection, MoELayer1, MoELayer2, MoELayer3, OutputLayer }
```

### Hybrid Architecture
```csharp
layers: { DenseLayer, MoELayer, DenseLayer, MoELayer, OutputLayer }
```

All use the same NeuralNetworkArchitecture + NeuralNetworkModel + PredictionModelBuilder pattern!
