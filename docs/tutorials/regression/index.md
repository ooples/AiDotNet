# Regression Tutorial

Learn how to predict continuous values using AiDotNet's regression algorithms.

---

## Overview

Regression is a supervised learning technique used to predict continuous numerical values. AiDotNet provides 41+ regression algorithms to choose from.

## Available Algorithms

| Algorithm | Use Case |
|:----------|:---------|
| Linear Regression | Simple linear relationships |
| Ridge Regression | Linear with L2 regularization |
| Lasso Regression | Linear with L1 regularization (feature selection) |
| ElasticNet | Combined L1/L2 regularization |
| Polynomial Regression | Non-linear relationships |
| Support Vector Regression | Complex non-linear patterns |
| Random Forest Regressor | Ensemble method, handles non-linearity |
| Gradient Boosting Regressor | High accuracy ensemble |
| Neural Network Regressor | Deep learning for complex patterns |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Regression;

// Sample data: predicting house prices
var features = new double[][]
{
    new[] { 1500.0, 3.0, 2.0 },  // sqft, bedrooms, bathrooms
    new[] { 2000.0, 4.0, 2.5 },
    new[] { 1200.0, 2.0, 1.0 }
};
var prices = new double[] { 300000, 450000, 200000 };

// Build and train
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestRegressor<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, prices);

// Predict
var predictedPrice = result.Predict(new[] { 1800.0, 3.0, 2.0 });
Console.WriteLine($"Predicted price: ${predictedPrice:N0}");
```

## Next Steps

- [Classification Tutorial](../classification/index.md) - Predict categories
- [Time Series Tutorial](../time-series/index.md) - Forecast future values
- [API Reference](../../api/index.md) - Full documentation
