---
title: "ML Basics"
description: "Learn the fundamentals of training models with AiModelBuilder."
order: 1
section: "Examples"
---


This guide demonstrates the fundamentals of using AiDotNet for machine learning tasks through the `AiModelBuilder` facade.

## Overview

Every model in AiDotNet is built the same way: `ConfigureModel(...)` + `ConfigureDataLoader(...)` + `BuildAsync()`, then `result.Predict(...)`. Metrics for the trained model are computed for you and read off the returned `AiModelResult` — through `result.GetDataSetStats(X, y)` for a specific dataset, or `result.Evaluation` for the internal train/validation/test split.

## Quick Start: Linear Regression

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Your training data (one feature per sample).
double[][] features =
{
    new[] { 1.0 }, new[] { 2.0 }, new[] { 3.0 }, new[] { 4.0 }, new[] { 5.0 }
};
double[] targets = { 2.1, 4.0, 5.9, 8.1, 10.0 };

var X = ToMatrix(features);
var y = new Vector<double>(targets);

// Build and train a model through the facade.
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

// Make a prediction for x = 6 (one-row matrix in, vector out).
var sixInput = new Matrix<double>(1, 1);
sixInput[0, 0] = 6.0;
Console.WriteLine($"Prediction for x=6: {result.Predict(sixInput)[0]:F2}");

// View model performance — no hand-rolled math.
var stats = result.GetDataSetStats(X, y);
Console.WriteLine($"R-Squared: {stats.PredictionStats.R2:F4}");
Console.WriteLine($"Mean Squared Error: {stats.ErrorStats.MSE:F4}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Multiple Features (Ridge Regression)

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// House price prediction: [sqft, bedrooms, age]
double[][] houseFeatures =
{
    new[] { 1500.0, 3.0, 10.0 }, new[] { 2000.0, 4.0, 5.0 },
    new[] { 1200.0, 2.0, 20.0 }, new[] { 1800.0, 3.0, 8.0 },
    new[] { 2500.0, 5.0, 2.0 }
};
double[] prices = { 300000, 450000, 200000, 380000, 550000 };

var X = ToMatrix(houseFeatures);
var y = new Vector<double>(prices);

// Ridge regression adds L2 regularization; swap in LassoRegression or
// ElasticNetRegression the same way.
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RidgeRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

var newHouse = new Matrix<double>(1, 3);
foreach (var (v, j) in new[] { 1700.0, 3.0, 12.0 }.Select((v, j) => (v, j)))
    newHouse[0, j] = v;
Console.WriteLine($"Estimated price: ${result.Predict(newHouse)[0]:N0}");

// Feature importance is available on the result.
Console.WriteLine("\nFeature Importance:");
foreach (var (name, importance) in result.GetFeatureImportance())
    Console.WriteLine($"  {name}: {importance:F3}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Binary Classification

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Customer churn: [tenure, monthly charges, has contract]
double[][] customerData =
{
    new[] { 12.0, 50.0, 1.0 }, new[] { 1.0, 80.0, 0.0 },
    new[] { 24.0, 45.0, 1.0 }, new[] { 3.0, 95.0, 0.0 },
    new[] { 36.0, 40.0, 1.0 }
};
double[] churned = { 0, 1, 0, 1, 0 };

var X = ToMatrix(customerData);
var y = new Vector<double>(churned);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

// Predict churn for a new customer.
var newCustomer = new Matrix<double>(1, 3);
foreach (var (v, j) in new[] { 6.0, 70.0, 0.0 }.Select((v, j) => (v, j)))
    newCustomer[0, j] = v;
Console.WriteLine($"Predicted churn class: {(int)result.Predict(newCustomer)[0]}");

// Classification metrics — ErrorStats auto-selects accuracy/precision/recall/F1.
var stats = result.GetDataSetStats(X, y);
Console.WriteLine($"Accuracy:  {stats.ErrorStats.Accuracy:P2}");
Console.WriteLine($"Precision: {stats.ErrorStats.Precision:P2}");
Console.WriteLine($"Recall:    {stats.ErrorStats.Recall:P2}");
Console.WriteLine($"F1 Score:  {stats.ErrorStats.F1Score:P2}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Multi-Class Classification

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Iris flowers: 4 measurements, 3 species (0=setosa, 1=versicolor, 2=virginica)
double[][] irisFeatures =
{
    new[] { 5.1, 3.5, 1.4, 0.2 }, new[] { 4.9, 3.0, 1.4, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 }, new[] { 6.4, 3.2, 4.5, 1.5 },
    new[] { 6.3, 3.3, 6.0, 2.5 }, new[] { 5.8, 2.7, 5.1, 1.9 }
};
double[] species = { 0, 0, 1, 1, 2, 2 };

var X = ToMatrix(irisFeatures);
var y = new Vector<double>(species);

// The classifier infers the number of classes from the labels.
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

var newFlower = new Matrix<double>(1, 4);
foreach (var (v, j) in new[] { 5.9, 3.0, 5.1, 1.8 }.Select((v, j) => (v, j)))
    newFlower[0, j] = v;
Console.WriteLine($"Predicted species: {(int)result.Predict(newFlower)[0]}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Reading Metrics for Train, Validation, and Test

The facade splits your data internally and evaluates the model on each part. Read it off `result.Evaluation` — no need to re-pass any data — to spot overfitting at a glance.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 },
    new[] { 7.0, 8.0 }, new[] { 8.0, 9.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(ToMatrix(features), new Vector<double>(targets)))
    .BuildAsync();

var evaluation = result.Evaluation;
Console.WriteLine($"Training R-Squared:   {evaluation.TrainingSet.PredictionStats.R2:F4}");
Console.WriteLine($"Validation R-Squared: {evaluation.ValidationSet.PredictionStats.R2:F4}");

// If you add .ConfigureCrossValidation(...), fold results land on result.CrossValidationResult.

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Saving and Loading Models

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 },
    new[] { 7.0, 8.0, 9.0 }, new[] { 10.0, 11.0, 12.0 }
};
double[] targets = { 10.0, 20.0, 30.0, 40.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(ToMatrix(features), new Vector<double>(targets)))
    .BuildAsync();

result.SaveModel("my_model.aimodel");
Console.WriteLine("Model saved!");

// Load it back, supplying a factory that constructs the right model type.
var loaded = AiModelResult<double, Matrix<double>, Vector<double>>.LoadModel(
    "my_model.aimodel",
    metadata => new MultipleRegression<double>());

var newData = new Matrix<double>(1, 3);
foreach (var (v, j) in new[] { 1.0, 2.0, 3.0 }.Select((v, j) => (v, j)))
    newData[0, j] = v;
Console.WriteLine($"Prediction: {loaded.Predict(newData)[0]:F2}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Batch Predictions

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] trainingFeatures =
{
    new[] { 1.0 }, new[] { 2.0 }, new[] { 3.0 }, new[] { 4.0 }, new[] { 5.0 }
};
double[] trainingTargets = { 2.0, 4.0, 6.0, 8.0, 10.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(ToMatrix(trainingFeatures), new Vector<double>(trainingTargets)))
    .BuildAsync();

// Predict many samples at once by passing a multi-row matrix.
double[][] testData = { new[] { 1.5 }, new[] { 2.5 }, new[] { 3.5 }, new[] { 4.5 } };
var predictions = result.Predict(ToMatrix(testData));

Console.WriteLine("Batch Predictions:");
for (int i = 0; i < predictions.Length; i++)
    Console.WriteLine($"  Input {testData[i][0]}: {predictions[i]:F2}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Model Comparison

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0 }, new[] { 2.0 }, new[] { 3.0 }, new[] { 4.0 }, new[] { 5.0 },
    new[] { 6.0 }, new[] { 7.0 }, new[] { 8.0 }, new[] { 9.0 }, new[] { 10.0 }
};
double[] targets = { 2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 14.0, 15.9, 18.1, 20.0 };

var X = ToMatrix(features);
var y = new Vector<double>(targets);

// Compare different regression models by swapping the ConfigureModel argument.
var models = new (string name, Func<IFullModel<double, Matrix<double>, Vector<double>>> make)[]
{
    ("Multiple",   () => new MultipleRegression<double>()),
    ("Ridge",      () => new RidgeRegression<double>()),
    ("Lasso",      () => new LassoRegression<double>()),
    ("ElasticNet", () => new ElasticNetRegression<double>()),
};

Console.WriteLine("Model Comparison (R-Squared):");
foreach (var (name, make) in models)
{
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(make())
        .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
        .BuildAsync();

    Console.WriteLine($"  {name,-12}: {result.GetDataSetStats(X, y).PredictionStats.R2:F4}");
}

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Summary

AiDotNet's `AiModelBuilder` provides:

- One fluent pattern for every model — `ConfigureModel` + `ConfigureDataLoader` + `BuildAsync`
- Rich metrics on the result — `result.GetDataSetStats(X, y)` and `result.Evaluation`
- Feature importance via `result.GetFeatureImportance()`
- Model saving and loading (`SaveModel` / `LoadModel`)
- A consistent `result.Predict(...)` for single and batch inputs
