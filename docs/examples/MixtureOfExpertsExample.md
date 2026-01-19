# Mixture-of-Experts (MoE) Usage Guide

This guide demonstrates how to use the Mixture-of-Experts neural network model with AiDotNet's AiModelBuilder.

## Overview

Mixture-of-Experts (MoE) is a neural network architecture that employs multiple specialist networks (experts) with learned routing. It enables models with extremely high capacity while remaining computationally efficient by activating only a subset of parameters per input.

## Standard Pattern (Same as All AiDotNet Models)

MoE follows the exact same pattern as other models in AiDotNet (like ARIMAModel, NBEATSModel, FeedForwardNeuralNetwork, etc.):

```csharp
using AiDotNet;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

// 1. Create configuration options
var options = new MixtureOfExpertsOptions<float>
{
    NumExperts = 8,                    // 8 specialist networks
    TopK = 2,                          // Use top 2 experts per input
    InputDim = 128,                    // Input dimension
    OutputDim = 128,                   // Output dimension
    HiddenExpansion = 4,               // 4x hidden layer expansion
    UseLoadBalancing = true,           // Enable load balancing
    LoadBalancingWeight = 0.01         // Load balancing loss weight
};

// 2. Create network architecture
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: 128,
    outputSize: 10
);

// 3. Create the model (implements IFullModel automatically)
var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

// 4. Use with AiModelBuilder (same as always)
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = builder
    .ConfigureModel(model)
    .Build(trainingData, trainingLabels);

// 5. Make predictions (same as always)
var predictions = builder.Predict(testData, result);
```

## Complete Example: Classification Task

```csharp
using AiDotNet;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

// Prepare your data
int numSamples = 1000;
int numFeatures = 784;  // e.g., 28x28 images flattened
int numClasses = 10;

var trainingData = new Tensor<float>(new[] { numSamples, numFeatures });
var trainingLabels = new Tensor<float>(new[] { numSamples, numClasses });
// ... fill with actual data ...

// Configure MoE model
var options = new MixtureOfExpertsOptions<float>
{
    NumExperts = 8,
    TopK = 2,
    InputDim = numFeatures,
    OutputDim = numFeatures,
    HiddenExpansion = 4,
    UseLoadBalancing = true,
    LoadBalancingWeight = 0.01
};

// Create architecture
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: numFeatures,
    outputSize: numClasses
);

// Create model
var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

// Train with AiModelBuilder
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
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

## Configuration Options

The `MixtureOfExpertsOptions` class provides all MoE-specific configuration:

### NumExperts
Controls how many specialist networks the model contains.

```csharp
options.NumExperts = 8;  // Default: 4
```

**Guidelines:**
- 2-4: Small models, limited compute
- 4-8: Most applications (recommended)
- 8-16: Larger, more complex tasks
- 16+: Very large models (use with TopK routing)

### TopK
Determines how many experts process each input (sparse routing).

```csharp
options.TopK = 2;  // Default: 2
```

**Guidelines:**
- `TopK = 1`: Only best expert - very fast, for 32+ experts
- `TopK = 2`: Top 2 experts - good balance for 8-32 experts (recommended)
- `TopK = 4`: More experts per input - higher quality but slower

### InputDim / OutputDim
Dimensions for expert networks.

```csharp
options.InputDim = 128;   // Default: 128
options.OutputDim = 128;  // Default: 128
```

**Guidelines:**
- Set InputDim to match your input features or previous layer output
- Often set OutputDim equal to InputDim for symmetry
- Use different values if you want to compress/expand representations

### HiddenExpansion
Controls the hidden layer size within each expert (as a multiple of InputDim).

```csharp
options.HiddenExpansion = 4;  // Default: 4 (from Transformer research)
```

**Guidelines:**
- 4: Standard choice (recommended, proven in research)
- 2-3: More efficient, less capacity
- 6-8: More capacity, higher compute cost

### UseLoadBalancing / LoadBalancingWeight
Controls auxiliary loss to ensure balanced expert usage.

```csharp
options.UseLoadBalancing = true;       // Default: true
options.LoadBalancingWeight = 0.01;    // Default: 0.01
```

**Guidelines:**
- Nearly always keep UseLoadBalancing = true
- LoadBalancingWeight 0.01 works well (gentle encouragement)
- Increase to 0.05-0.1 if you notice severe expert imbalance
- Decrease to 0.001 if training seems unstable

### RandomSeed
Controls reproducibility of initialization.

```csharp
options.RandomSeed = 42;  // Default: null (non-deterministic)
```

Set a specific value for reproducible results (useful for research and debugging).

## Regression Example

MoE works for regression too - just change the task type:

```csharp
// Configure MoE for regression
var options = new MixtureOfExpertsOptions<float>
{
    NumExperts = 4,
    TopK = 2,
    InputDim = 10,
    OutputDim = 10
};

var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.Regression,  // Regression task
    inputSize: 10,
    outputSize: 1
);

var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);
var result = builder.ConfigureModel(model).Build(trainingData, trainingTargets);
```

## Advanced: Custom Layer Architecture

For more control, you can provide custom layers in the architecture:

```csharp
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

// Create custom layers including MoE layer
var moeLayer = new MixtureOfExpertsBuilder<float>()
    .WithExperts(8)
    .WithDimensions(256, 256)
    .WithTopK(2)
    .WithLoadBalancing(true)
    .Build();

var layers = new List<ILayer<float>>
{
    new DenseLayer<float>(784, 256, new ReLUActivation<float>()),
    moeLayer,
    new DenseLayer<float>(256, 10, new SoftmaxActivation<float>())
};

// Pass custom layers to architecture
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: 784,
    outputSize: 10,
    layers: layers  // Provide custom layers
);

// Create model with custom architecture
var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);
```

**Note:** When providing custom layers, the options are only used for metadata. The actual MoE layer comes from your custom layers list.

## Comparison with Other AiDotNet Models

MoE follows the exact same pattern as other models:

### Time Series Models
```csharp
// ARIMA Model
var arimaOptions = new ARIMAOptions<float> { P = 2, D = 1, Q = 2 };
var arimaModel = new ARIMAModel<float>(arimaOptions);
var result = builder.ConfigureModel(arimaModel).Build(data, labels);

// MoE Model (same pattern!)
var moeOptions = new MixtureOfExpertsOptions<float> { NumExperts = 8, TopK = 2 };
var moeModel = new MixtureOfExpertsNeuralNetwork<float>(moeOptions, architecture);
var result = builder.ConfigureModel(moeModel).Build(data, labels);
```

### Neural Network Models
```csharp
// Feed-Forward Network
var ffnn = new FeedForwardNeuralNetwork<float>(architecture);
var result = builder.ConfigureModel(ffnn).Build(data, labels);

// MoE Network (same pattern!)
var moeModel = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);
var result = builder.ConfigureModel(moeModel).Build(data, labels);
```

## Monitoring Expert Usage

You can monitor how balanced expert usage is during training:

```csharp
// After training, get diagnostics
var metadata = model.GetModelMetadata();

Console.WriteLine("\nModel Information:");
Console.WriteLine($"Number of Experts: {metadata.AdditionalInfo["NumExperts"]}");
Console.WriteLine($"TopK: {metadata.AdditionalInfo["TopK"]}");
Console.WriteLine($"Load Balancing Enabled: {metadata.AdditionalInfo["UseLoadBalancing"]}");

// For more detailed diagnostics, access the underlying MoE layer
if (model is MixtureOfExpertsNeuralNetwork<float> moeNet)
{
    // Access layer-level diagnostics as needed
    // (implementation depends on exposing layer diagnostics)
}
```

## Best Practices

1. **Start Simple**: Begin with 4-8 experts and TopK=2, then scale up if needed

2. **Match Dimensions**: Set InputDim to match your input features:
   ```csharp
   options.InputDim = yourInputSize;
   ```

3. **Keep Load Balancing On**: Nearly always use load balancing to prevent expert collapse:
   ```csharp
   options.UseLoadBalancing = true;
   options.LoadBalancingWeight = 0.01;  // Gentle but effective
   ```

4. **Use Standard Pattern**: Follow the same pattern as all AiDotNet models:
   - Create Options → Create Architecture → Create Model → Use with AiModelBuilder

5. **Monitor Training**: Check that training loss decreases and validation accuracy improves

6. **Scale Appropriately**: Use more experts for complex tasks, fewer for simpler ones

## Key Points

1. **Configuration Class**: MoE uses `MixtureOfExpertsOptions` for all configuration
2. **Implements IFullModel**: Works automatically with AiModelBuilder
3. **Same Pattern**: Identical to ARIMAModel, NBEATSModel, FeedForwardNeuralNetwork, etc.
4. **No Special Helpers**: No extension methods or special builders needed at model level
5. **Automatic Integration**: Load balancing loss is automatically integrated during training

## Common Use Cases

### Image Classification
```csharp
var options = new MixtureOfExpertsOptions<float>
{
    NumExperts = 8,
    TopK = 2,
    InputDim = 784,
    OutputDim = 784
};
```

### Natural Language Processing
```csharp
var options = new MixtureOfExpertsOptions<float>
{
    NumExperts = 16,
    TopK = 2,
    InputDim = 512,
    OutputDim = 512
};
```

### Tabular Data
```csharp
var options = new MixtureOfExpertsOptions<float>
{
    NumExperts = 4,
    TopK = 2,
    InputDim = numFeatures,
    OutputDim = numFeatures
};
```

## Summary

Mixture-of-Experts in AiDotNet follows the standard model pattern:

1. Create a `MixtureOfExpertsOptions<T>` configuration object
2. Create a `NeuralNetworkArchitecture<T>` defining the task
3. Create a `MixtureOfExpertsNeuralNetwork<T>` model
4. Use with `AiModelBuilder` for training and inference

This is the same pattern as all other models in AiDotNet, making it easy to use and integrate into your workflows.
