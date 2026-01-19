---
layout: default
title: Quick Start
parent: Getting Started
nav_order: 2
---

# Quick Start Tutorial
{: .no_toc }

Build your first AI model with AiDotNet in 5 minutes.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Create a New Project

```bash
dotnet new console -n MyFirstAiModel
cd MyFirstAiModel
dotnet add package AiDotNet
```

## Example 1: Classification

Classify iris flowers into species:

```csharp
using AiDotNet;
using AiDotNet.Classification;

// Training data (Iris dataset)
var features = new double[][]
{
    // Setosa
    new[] { 5.1, 3.5, 1.4, 0.2 },
    new[] { 4.9, 3.0, 1.4, 0.2 },
    // Versicolor
    new[] { 7.0, 3.2, 4.7, 1.4 },
    new[] { 6.4, 3.2, 4.5, 1.5 },
    // Virginica
    new[] { 6.3, 3.3, 6.0, 2.5 },
    new[] { 5.8, 2.7, 5.1, 1.9 }
};
var labels = new double[] { 0, 0, 1, 1, 2, 2 };

// Build classifier
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

// Make prediction using result object (facade pattern)
var sample = new[] { 5.9, 3.0, 5.1, 1.8 };
var prediction = result.Predict(sample);

Console.WriteLine($"Predicted species: {prediction}");  // Output: 2 (Virginica)
```

## Example 2: Regression

Predict house prices:

```csharp
using AiDotNet;
using AiDotNet.Regression;

// Features: sqft, bedrooms, bathrooms
var features = new double[][]
{
    new[] { 1500.0, 3.0, 2.0 },
    new[] { 2000.0, 4.0, 2.5 },
    new[] { 1200.0, 2.0, 1.0 },
    new[] { 2500.0, 4.0, 3.0 }
};
var prices = new double[] { 300000, 450000, 200000, 550000 };

// Build regressor
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingRegression<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, prices);

// Predict price using result object (facade pattern)
var newHouse = new[] { 1800.0, 3.0, 2.0 };
var predictedPrice = result.Predict(newHouse);

Console.WriteLine($"Predicted price: ${predictedPrice:N0}");
```

## Example 3: Neural Network

Build a simple neural network:

```csharp
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.LinearAlgebra;

// XOR problem
var features = new Tensor<double>(new int[] { 4, 2 });
var labels = new Tensor<double>(new int[] { 4, 1 });

// Set XOR data
double[,] xorInputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
double[] xorOutputs = { 0, 1, 1, 0 };

for (int i = 0; i < 4; i++)
{
    for (int j = 0; j < 2; j++)
        features[new[] { i, j }] = xorInputs[i, j];
    labels[new[] { i, 0 }] = xorOutputs[i];
}

// Build neural network
var network = new NeuralNetwork<double>(
    new NeuralNetworkArchitecture<double>(
        inputFeatures: 2,
        numClasses: 1,
        complexity: NetworkComplexity.Simple));

// Train
for (int epoch = 0; epoch < 1000; epoch++)
{
    network.Train(features, labels);

    if (epoch % 200 == 0)
        Console.WriteLine($"Epoch {epoch}: Loss = {network.GetLastLoss():F4}");
}

// Predict
var predictions = network.Predict(features);
Console.WriteLine($"Prediction for [1,0]: {predictions[new[] { 2, 0 }]:F2}");
```

## Example 4: With Cross-Validation

Add cross-validation to evaluate model performance:

```csharp
using AiDotNet;
using AiDotNet.Classification;
using AiDotNet.CrossValidation;

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .BuildAsync(features, labels);

// Access CV results
if (result.CrossValidationResult != null)
{
    var cv = result.CrossValidationResult;
    Console.WriteLine($"Mean Accuracy: {cv.MeanScore:P2}");
    Console.WriteLine($"Std Dev: {cv.StandardDeviation:P2}");
}
```

## Example 5: With GPU Acceleration

Enable GPU for faster training:

```csharp
using AiDotNet;
using AiDotNet.Configuration;

var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(cnnModel)
    .ConfigureOptimizer(new AdamOptimizer<float>())
    .ConfigureGpuAcceleration(new GpuAccelerationConfig
    {
        Enabled = true,
        DeviceId = 0
    })
    .BuildAsync(trainImages, trainLabels);
```

## Next Steps

Now that you've built your first models, explore:

- [Core Concepts](./concepts) - Understand the architecture
- [Tutorials](/tutorials/) - Task-specific guides
- [Samples](/samples/) - Complete working examples
- [API Reference](/api/) - Full API documentation
