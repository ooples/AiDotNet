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

Every model is built the same way: `ConfigureModel(...)` + `ConfigureDataLoader(...)` + `BuildAsync()`, then `result.Predict(...)`.

## Example 1: Classification

Classify iris flowers into species:

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 5.1, 3.5, 1.4, 0.2 }, new[] { 4.9, 3.0, 1.4, 0.2 },  // Setosa
    new[] { 7.0, 3.2, 4.7, 1.4 }, new[] { 6.4, 3.2, 4.5, 1.5 },  // Versicolor
    new[] { 6.3, 3.3, 6.0, 2.5 }, new[] { 5.8, 2.7, 5.1, 1.9 }   // Virginica
};
double[] labels = { 0, 0, 1, 1, 2, 2 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureDataLoader(DataLoaders.FromArrays(features, labels))
    .BuildAsync();

var newFlower = new Matrix<double>(1, 4);
foreach (var (v, j) in new[] { 5.9, 3.0, 5.1, 1.8 }.Select((v, j) => (v, j)))
    newFlower[0, j] = v;
Console.WriteLine($"Predicted species: {(int)result.Predict(newFlower)[0]}");
```

## Example 2: Regression

Predict house prices:

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1500.0, 3.0, 2.0 }, new[] { 2000.0, 4.0, 2.5 },
    new[] { 1200.0, 2.0, 1.0 }, new[] { 2500.0, 4.0, 3.0 }
};
double[] prices = { 300000, 450000, 200000, 550000 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GradientBoostingRegression<double>(
        new GradientBoostingRegressionOptions { NumberOfTrees = 100 }))
    .ConfigureDataLoader(DataLoaders.FromArrays(features, prices))
    .BuildAsync();

var newHouse = new Matrix<double>(1, 3);
foreach (var (v, j) in new[] { 1800.0, 3.0, 2.0 }.Select((v, j) => (v, j)))
    newHouse[0, j] = v;
Console.WriteLine($"Predicted price: ${result.Predict(newHouse)[0]:N0}");
```

## Example 3: Neural Network

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

// 6 iris samples, 4 features, one-hot labels for 3 species.
var trainX = new Tensor<double>(new[] { 6, 4 });
var trainY = new Tensor<double>(new[] { 6, 3 });
double[][] rows =
{
    new[] { 5.1, 3.5, 1.4, 0.2 }, new[] { 4.9, 3.0, 1.4, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 }, new[] { 6.4, 3.2, 4.5, 1.5 },
    new[] { 6.3, 3.3, 6.0, 2.5 }, new[] { 5.8, 2.7, 5.1, 1.9 }
};
int[] species = { 0, 0, 1, 1, 2, 2 };
for (int i = 0; i < 6; i++)
{
    for (int j = 0; j < 4; j++) trainX[new[] { i, j }] = rows[i][j];
    trainY[new[] { i, species[i] }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 4, numClasses: 3, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Output shape: [{string.Join(", ", result.Predict(trainX).Shape)}]");
```

## Example 4: With Cross-Validation

```csharp
using AiDotNet;
using AiDotNet.CrossValidators;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1500.0, 3.0 }, new[] { 2000.0, 4.0 }, new[] { 1200.0, 2.0 },
    new[] { 2500.0, 4.0 }, new[] { 1800.0, 3.0 }, new[] { 2200.0, 4.0 },
    new[] { 1400.0, 2.0 }, new[] { 2600.0, 5.0 }, new[] { 1700.0, 3.0 }, new[] { 2100.0, 4.0 }
};
double[] prices = { 300000, 450000, 200000, 550000, 380000, 480000, 240000, 600000, 360000, 470000 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GradientBoostingRegression<double>(
        new GradientBoostingRegressionOptions { NumberOfTrees = 50 }))
    .ConfigureCrossValidation(new KFoldCrossValidator<double, Matrix<double>, Vector<double>>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, prices))
    .BuildAsync();

if (result.CrossValidationResult is not null)
{
    Console.WriteLine($"Mean R²:  {result.CrossValidationResult.R2Stats.Mean:F4}");
    Console.WriteLine($"Std Dev:  {result.CrossValidationResult.R2Stats.StandardDeviation:F4}");
}
```

## Example 5: With GPU Acceleration

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(0);
var trainX = new Tensor<float>(new[] { 64, 32 });
var trainY = new Tensor<float>(new[] { 64, 4 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 32; j++) trainX[new[] { i, j }] = (float)rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1f;
}

var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple));

// ConfigureGpuAcceleration enables the GPU path when a device is available
// and transparently falls back to CPU otherwise.
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureGpuAcceleration()
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Trained; output shape [{string.Join(", ", result.Predict(trainX).Shape)}]");
```
