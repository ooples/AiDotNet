---
title: "Neural Network Training"
description: "Train a neural network from scratch."
order: 2
section: "Examples"
---


This guide demonstrates how to train neural networks for various tasks using AiDotNet's `AiModelBuilder` facade. You configure a model and a data loader, call `BuildAsync()`, and predict through the returned `AiModelResult`.

## Overview

Every model in AiDotNet is built the same way: `ConfigureModel(...)` + `ConfigureDataLoader(...)` + `BuildAsync()`, then `result.Predict(...)`.

## Image Classification

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// 100 training samples of 784 features (28x28), one-hot labels for 10 digits.
// In production, load MNIST through a DataLoader instead of synthesising data.
var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 100, 784 });
var trainY = new Tensor<double>(new[] { 100, 10 });
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < 784; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 10 }] = 1.0;
}

// The architecture builds an appropriate network from the input/class counts.
var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 784, numClasses: 10, complexity: NetworkComplexity.Medium));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

// Predict class scores for a batch (here, the training batch) and read row 0.
var scores = result.Predict(trainX);
int predicted = 0;
for (int c = 1; c < 10; c++)
    if (scores[new[] { 0, c }] > scores[new[] { 0, predicted }]) predicted = c;
Console.WriteLine($"Predicted digit for sample 0: {predicted}");
```

## Binary Classification (Spam Detection)

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Email features (word frequencies, etc.) and spam labels (1 = spam).
double[][] emailFeatures =
{
    new[] { 0.1, 0.8, 0.0, 0.9 },
    new[] { 0.9, 0.1, 0.7, 0.0 },
    new[] { 0.2, 0.7, 0.1, 0.8 },
    new[] { 0.3, 0.6, 0.2, 0.7 }
};
double[] isSpam = { 1, 0, 1, 1 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureDataLoader(DataLoaders.FromArrays(emailFeatures, isSpam))
    .BuildAsync();

// Classify a new email (one-row matrix in, vector of class scores out).
var newEmail = new Matrix<double>(1, 4);
foreach (var (v, j) in new[] { 0.3, 0.6, 0.2, 0.7 }.Select((v, j) => (v, j)))
    newEmail[0, j] = v;
Console.WriteLine($"Spam class: {(int)Math.Round(result.Predict(newEmail)[0])}");
```

## Regression (Price Prediction)

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] productFeatures =
{
    new[] { 100.0, 4.5, 1000.0, 2.0 },
    new[] { 250.0, 4.2, 500.0, 1.0 },
    new[] { 50.0, 4.8, 2000.0, 3.0 },
    new[] { 150.0, 3.9, 750.0, 2.0 }
};
double[] prices = { 29.99, 59.99, 19.99, 39.99 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GradientBoostingRegression<double>(
        new GradientBoostingRegressionOptions { NumberOfTrees = 100, MaxDepth = 3 }))
    .ConfigureDataLoader(DataLoaders.FromArrays(productFeatures, prices))
    .BuildAsync();

var newProduct = new Matrix<double>(1, 4);
foreach (var (v, j) in new[] { 150.0, 4.3, 750.0, 2.0 }.Select((v, j) => (v, j)))
    newProduct[0, j] = v;
Console.WriteLine($"Predicted price: ${result.Predict(newProduct)[0]:F2}");
```

## Time Series Forecasting

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Each row is a 5-day window of sales; the target is the next day's sales.
double[][] windows =
{
    new[] { 100.0, 110.0, 105.0, 115.0, 120.0 },
    new[] { 110.0, 105.0, 115.0, 120.0, 125.0 },
    new[] { 105.0, 115.0, 120.0, 125.0, 130.0 },
    new[] { 115.0, 120.0, 125.0, 130.0, 135.0 }
};
double[] nextDay = { 125, 130, 135, 140 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GradientBoostingRegression<double>(
        new GradientBoostingRegressionOptions { NumberOfTrees = 100 }))
    .ConfigureDataLoader(DataLoaders.FromArrays(windows, nextDay))
    .BuildAsync();

var recent = new Matrix<double>(1, 5);
foreach (var (v, j) in new[] { 125.0, 130.0, 135.0, 140.0, 145.0 }.Select((v, j) => (v, j)))
    recent[0, j] = v;
Console.WriteLine($"Forecasted sales: ${result.Predict(recent)[0]:F0}");
```

## Multi-Class Classification with an Optimizer

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(7);
var x = new Tensor<double>(new[] { 200, 64 });
var y = new Tensor<double>(new[] { 200, 10 });
for (int i = 0; i < 200; i++)
{
    for (int j = 0; j < 64; j++) x[new[] { i, j }] = rng.NextDouble();
    y[new[] { i, i % 10 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 64, numClasses: 10, complexity: NetworkComplexity.Simple));

// ConfigureOptimizer plugs in a specific optimizer; other knobs (LR schedule,
// early stopping, regularization) have their own Configure* methods.
var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
    .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .BuildAsync();

var preds = result.Predict(x);
Console.WriteLine($"Output shape: [{string.Join(", ", preds.Shape)}]");
```

## GPU Acceleration

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(1);
var x = new Tensor<double>(new[] { 64, 32 });
var y = new Tensor<double>(new[] { 64, 3 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 32; j++) x[new[] { i, j }] = rng.NextDouble();
    y[new[] { i, i % 3 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 32, numClasses: 3, complexity: NetworkComplexity.Simple));

// ConfigureGpuAcceleration enables the GPU path when a device is available;
// it transparently falls back to CPU otherwise.
var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
    .ConfigureGpuAcceleration()
    .BuildAsync();

Console.WriteLine($"Trained; output shape [{string.Join(", ", result.Predict(x).Shape)}]");
```

## Reading Training Metrics

Everything you need to evaluate a trained model hangs off the returned `AiModelResult`.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(3);
var x = new Tensor<double>(new[] { 120, 16 });
var y = new Tensor<double>(new[] { 120, 4 });
for (int i = 0; i < 120; i++)
{
    for (int j = 0; j < 16; j++) x[new[] { i, j }] = rng.NextDouble();
    y[new[] { i, i % 4 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
        inputFeatures: 16, numClasses: 4, complexity: NetworkComplexity.Simple)))
    .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
    .BuildAsync();

// Model summary.
Console.WriteLine($"Layers: {result.LayerCount}, trainable params: {result.TotalTrainableParameters:N0}");

// Per-epoch training metrics (loss, accuracy, …) keyed by metric name, when recorded.
var history = result.GetTrainingMetricsHistory();
if (history is not null)
    foreach (var (metric, values) in history)
        Console.WriteLine($"{metric}: final = {values[^1]:F4} (over {values.Count} epochs)");

// If you add .ConfigureCrossValidation(...), the fold results land on result.CrossValidationResult.
```

## Best Practices

1. **Start with small models**: Begin simple and increase complexity only if needed.
2. **Use validation data**: Always monitor validation metrics to detect overfitting.
3. **Normalize your data**: Neural networks train better with normalized inputs (`ConfigurePreprocessing`).
4. **Use early stopping**: `ConfigureStoppingCriterion` prevents overfitting.
5. **Experiment with learning rates**: Usually the most important hyperparameter (`ConfigureOptimizer`).
6. **Use data augmentation**: `ConfigureAugmentation`, especially for image tasks with limited data.

## Summary

AiDotNet's `AiModelBuilder` makes neural network training accessible: configure a model + a data loader, `BuildAsync()`, and predict through the returned `AiModelResult`. Cross-cutting concerns (optimizer, GPU, preprocessing, augmentation, early stopping) are added with their own `Configure*` methods.
