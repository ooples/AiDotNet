---
title: Quick Start
description: Build your first AI model with AiDotNet in 5 minutes.
order: 2
section: Getting Started
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

var features = new double[][]
{
    new[] { 5.1, 3.5, 1.4, 0.2 },  // Setosa
    new[] { 4.9, 3.0, 1.4, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 },  // Versicolor
    new[] { 6.4, 3.2, 4.5, 1.5 },
    new[] { 6.3, 3.3, 6.0, 2.5 },  // Virginica
    new[] { 5.8, 2.7, 5.1, 1.9 }
};
var labels = new double[] { 0, 0, 1, 1, 2, 2 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Predict(new[] { 5.9, 3.0, 5.1, 1.8 });
Console.WriteLine($"Predicted species: {prediction}");  // Output: 2 (Virginica)
```

## Example 2: Regression

Predict house prices:

```csharp
using AiDotNet;
using AiDotNet.Regression;

var features = new double[][]
{
    new[] { 1500.0, 3.0, 2.0 },
    new[] { 2000.0, 4.0, 2.5 },
    new[] { 1200.0, 2.0, 1.0 },
    new[] { 2500.0, 4.0, 3.0 }
};
var prices = new double[] { 300000, 450000, 200000, 550000 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingRegression<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, prices);

var predictedPrice = result.Predict(new[] { 1800.0, 3.0, 2.0 });
Console.WriteLine($"Predicted price: ${predictedPrice:N0}");
```

## Example 3: Neural Network

```csharp
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetwork<double>(
        inputSize: 4, hiddenSize: 16, outputSize: 3))
    .ConfigureOptimizer(new AdamOptimizer<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var newSample = new double[] { 5.1, 3.5, 1.4, 0.2 };
var prediction = result.Predict(newSample);
```

## Example 4: With Cross-Validation

```csharp
using AiDotNet;
using AiDotNet.Classification;
using AiDotNet.CrossValidation;

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .BuildAsync(features, labels);

if (result.CrossValidationResult != null)
{
    Console.WriteLine($"Mean Accuracy: {result.CrossValidationResult.MeanScore:P2}");
    Console.WriteLine($"Std Dev: {result.CrossValidationResult.StandardDeviation:P2}");
}
```

## Example 5: With GPU Acceleration

```csharp
using AiDotNet;
using AiDotNet.Optimizers;
using AiDotNet.ComputerVision;

var cnnModel = new ResNet18<float>();
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
