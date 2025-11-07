# Mixture-of-Experts (MoE) Usage Guide

This guide demonstrates how to use Mixture-of-Experts layers with AiDotNet's PredictionModelBuilder pattern.

## Overview

Mixture-of-Experts (MoE) is a powerful architecture that enables models with extremely high capacity while remaining computationally efficient by activating only a subset of parameters per input.

## Quick Start

The easiest way to use MoE is through the extension methods:

```csharp
using AiDotNet;
using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;

// 1. Prepare your data
var trainingData = new Tensor<float>(new[] { 100, 10 }); // 100 samples, 10 features
var trainingLabels = new Tensor<float>(new[] { 100, 3 }); // 3 classes

// 2. Create MoE model using extension method
var moeModel = MixtureOfExpertsExtensions.CreateMoEModel<float>(
    inputSize: 10,          // Number of input features
    outputSize: 3,          // Number of output classes
    numExperts: 8,          // 8 specialist networks
    topK: 2,                // Use top 2 experts per input
    useLoadBalancing: true  // Enable load balancing
);

// 3. Configure and build with PredictionModelBuilder
var builder = new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = builder
    .ConfigureModel(moeModel)
    .Build(trainingData, trainingLabels);

// 4. Make predictions
var testData = new Tensor<float>(new[] { 10, 10 }); // 10 test samples
var predictions = builder.Predict(testData, result);
```

## Advanced: Custom Architecture

For more control, create a custom architecture:

```csharp
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

// Create custom MoE architecture
var architecture = MixtureOfExpertsExtensions.CreateMoEArchitecture<float>(
    inputSize: 128,
    outputSize: 10,
    numExperts: 16,
    topK: 4,  // Use top 4 out of 16 experts
    useLoadBalancing: true,
    loadBalancingWeight: 0.01,
    taskType: NeuralNetworkTaskType.MultiClassClassification
);

// Wrap in model
var model = new NeuralNetworkModel<float>(architecture);

// Use with PredictionModelBuilder
var builder = new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = builder.ConfigureModel(model).Build(trainingData, trainingLabels);
```

## Deep MoE Architecture

For complex tasks requiring deep learning:

```csharp
// Create a 3-layer deep MoE model
var deepMoE = MixtureOfExpertsExtensions.CreateDeepMoEModel<float>(
    inputSize: 128,         // Input features
    hiddenSize: 256,        // Hidden layer size
    outputSize: 10,         // Output classes
    numMoELayers: 3,        // Stack 3 MoE layers
    numExperts: 8,          // 8 experts per layer
    topK: 2                 // Top 2 experts per layer
);

var result = new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(deepMoE)
    .Build(trainingData, trainingLabels);
```

## Manual Configuration (Maximum Control)

For complete control over the architecture:

```csharp
using AiDotNet.ActivationFunctions;

// 1. Build MoE layer using the builder
var moeLayer = new MixtureOfExpertsBuilder<float>()
    .WithExperts(8)                     // 8 experts
    .WithDimensions(128, 128)           // Input/output size
    .WithExpertHiddenDim(512)           // Expert hidden size (4x expansion)
    .WithTopK(2)                        // Sparse routing
    .WithLoadBalancing(true, 0.01)      // Load balancing enabled
    .WithExpertActivation(new GELUActivation<float>())  // Custom activation
    .Build();

// 2. Create custom layer sequence
var layers = new List<ILayer<float>>
{
    moeLayer,
    new DenseLayer<float>(128, 10, new SoftmaxActivation<float>())
};

// 3. Create architecture with custom layers
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.Medium,
    inputSize: 128,
    outputSize: 10,
    layers: layers
);

// 4. Wrap and use with PredictionModelBuilder
var model = new NeuralNetworkModel<float>(architecture);
var result = builder.ConfigureModel(model).Build(trainingData, trainingLabels);
```

## Monitoring Expert Usage

Check if experts are being used balanced during training:

```csharp
using AiDotNet.Interfaces;

// After training, access the underlying network
var network = ((NeuralNetworkModel<float>)result.Model).Network;

// Find MoE layers
foreach (var layer in network.Layers)
{
    if (layer is IAuxiliaryLossLayer<float> auxLayer)
    {
        var diagnostics = auxLayer.GetAuxiliaryLossDiagnostics();

        Console.WriteLine("Expert Usage Statistics:");
        foreach (var (key, value) in diagnostics)
        {
            Console.WriteLine($"{key}: {value}");
        }
    }
}
```

## Parameter Guidelines

### Number of Experts (`numExperts`)
- **2-4 experts**: Good for small models or limited compute
- **4-8 experts**: Sweet spot for most applications
- **8-16 experts**: For larger, more complex tasks
- **16+ experts**: For very large scale models (use with TopK)

### Top-K Selection (`topK`)
- **topK = 0**: All experts process every input (soft routing)
  - Best for: Small models (4-8 experts), maximum quality
- **topK = 1**: Only the best expert per input
  - Best for: Very large models (32+ experts), inference speed critical
- **topK = 2**: Top 2 experts per input
  - Best for: Medium to large models (8-32 experts), good balance

### Load Balancing Weight (`loadBalancingWeight`)
- **0.01**: Gentle encouragement (default, rarely hurts accuracy)
- **0.05**: Moderate encouragement (use if you see imbalance)
- **0.1**: Strong encouragement (may slightly reduce accuracy)

## Complete Example: Classification Task

```csharp
using AiDotNet;
using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

// Prepare data (e.g., MNIST-like classification)
int numSamples = 1000;
int numFeatures = 784;  // 28x28 images flattened
int numClasses = 10;    // 10 digits

var trainingData = new Tensor<float>(new[] { numSamples, numFeatures });
var trainingLabels = new Tensor<float>(new[] { numSamples, numClasses });

// ... fill with actual data ...

// Create MoE model with sensible defaults
var moeModel = MixtureOfExpertsExtensions.CreateMoEModel<float>(
    inputSize: numFeatures,
    outputSize: numClasses,
    numExperts: 8,
    topK: 2,  // Use only top 2 experts (75% computation reduction!)
    useLoadBalancing: true,
    taskType: NeuralNetworkTaskType.MultiClassClassification
);

// Build and train
var builder = new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = builder
    .ConfigureModel(moeModel)
    .Build(trainingData, trainingLabels);

// Evaluate
Console.WriteLine($"Training Accuracy: {result.TrainingAccuracy:P2}");
Console.WriteLine($"Validation Accuracy: {result.ValidationAccuracy:P2}");

// Make predictions
var testData = new Tensor<float>(new[] { 100, numFeatures });
// ... fill with test data ...

var predictions = builder.Predict(testData, result);

// Save for later use
builder.SaveModel(result, "moe_classifier.bin");

// Load and use later
var loadedModel = builder.LoadModel("moe_classifier.bin");
var newPredictions = builder.Predict(testData, loadedModel);
```

## Tips for Best Results

1. **Start Simple**: Begin with 4-8 experts and soft routing (topK=0)
2. **Add Sparsity**: Once working, try topK=2 for efficiency gains
3. **Monitor Balance**: Check expert usage diagnostics to ensure no collapse
4. **Tune Load Balancing**: Adjust weight if you see severe imbalance
5. **Scale Gradually**: Increase number of experts as data and compute allow

## Research References

- Switch Transformer (Google, 2021): Introduced load balancing loss
- GShard (Google, 2020): Demonstrated massive scaling with MoE
- Mixtral 8x7B (Mistral AI, 2023): Modern sparse MoE for language models

## Architecture Comparison

| Configuration | Experts | TopK | Parameters | Computation | Best For |
|--------------|---------|------|------------|-------------|----------|
| Dense Baseline | - | - | 100K | 100% | Small datasets |
| MoE Soft | 4 | 0 | 400K | 400% | Moderate datasets |
| MoE Sparse | 8 | 2 | 800K | 200% | Large datasets |
| MoE Very Sparse | 16 | 1 | 1.6M | 100% | Very large scale |

As shown, sparse MoE (TopK < numExperts) provides massive capacity with controlled computation!
