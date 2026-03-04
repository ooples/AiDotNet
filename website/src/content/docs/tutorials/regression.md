---
title: "Regression"
description: "Predict continuous values with AiDotNet."
order: 2
section: "Tutorials"
---

Learn how to predict continuous values using AiDotNet's regression algorithms.

## What is Regression?

Regression is a supervised learning task where the goal is to predict continuous numeric values from input features. Examples include:
- House price prediction
- Temperature forecasting
- Stock price estimation
- Sales revenue prediction

## Types of Regression

### Simple Regression
One input feature predicts one output. Example: Predict house price from square footage.

### Multiple Regression
Multiple input features predict one output. Example: Predict house price from square footage, bedrooms, and location.

### Polynomial Regression
Non-linear relationships using polynomial features.

---

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Regression;

// Prepare data (house features: sqft, bedrooms, bathrooms, age)
var features = new double[][]
{
    new[] { 1400.0, 3.0, 2.0, 15.0 },
    new[] { 1600.0, 3.0, 2.0, 10.0 },
    new[] { 1700.0, 3.0, 2.5, 5.0 },
    new[] { 1875.0, 4.0, 3.0, 8.0 },
    new[] { 1100.0, 2.0, 1.0, 25.0 },
    new[] { 2200.0, 4.0, 3.0, 2.0 }
};
var prices = new double[] { 245000, 312000, 279000, 308000, 199000, 425000 };

// Build and train
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestRegressor<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, prices);

// Predict
var newHome = new double[] { 1500.0, 3.0, 2.0, 12.0 };
var predictedPrice = result.Predict(newHome);
Console.WriteLine($"Predicted price: ${predictedPrice:N0}");
```

---

## Available Regressors

### Tree-Based Methods

| Regressor | Description | Best For |
|:----------|:------------|:---------|
| `RandomForestRegressor` | Ensemble of decision trees | General purpose, robust |
| `GradientBoostingRegressor` | Boosted decision trees | High accuracy |
| `DecisionTreeRegressor` | Single decision tree | Interpretability |

### Linear Methods

| Regressor | Description | Best For |
|:----------|:------------|:---------|
| `LinearRegression` | Ordinary least squares | Simple relationships |
| `RidgeRegression` | L2-regularized linear | Multicollinearity |
| `LassoRegression` | L1-regularized linear | Feature selection |
| `ElasticNetRegression` | L1+L2 regularization | Best of both |

### Distance-Based

| Regressor | Description | Best For |
|:----------|:------------|:---------|
| `KNearestNeighborsRegressor` | Instance-based learning | Non-linear, small datasets |

### Neural Networks

```csharp
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(new NeuralNetworkRegressor<float>(
        inputFeatures: 10,
        complexity: NetworkComplexity.Medium))
    .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 0.001f))
    .BuildAsync(features, targets);
```

---

## Data Preprocessing

### Automatic Preprocessing

```csharp
.ConfigurePreprocessing()  // Applies StandardScaler by default
```

### Custom Preprocessing

```csharp
.ConfigurePreprocessing(new PreprocessingConfig
{
    Scaler = new MinMaxScaler<double>(),
    ImputeStrategy = ImputeStrategy.Median,
    HandleCategorical = true
})
```

---

## Evaluation Metrics

```csharp
var predictions = testSamples.Select(s => result.Predict(s)).ToArray();

// Common regression metrics
Console.WriteLine($"MAE: {Metrics.MeanAbsoluteError(predictions, testTargets):F4}");
Console.WriteLine($"MSE: {Metrics.MeanSquaredError(predictions, testTargets):F4}");
Console.WriteLine($"RMSE: {Metrics.RootMeanSquaredError(predictions, testTargets):F4}");
Console.WriteLine($"R2: {Metrics.RSquared(predictions, testTargets):F4}");
```

---

## Best Practices

1. **Start simple**: Use Linear Regression as a baseline
2. **Check for outliers**: Outliers heavily influence linear models
3. **Feature scaling**: Most algorithms benefit from scaled features
4. **Regularize**: Use Ridge or Lasso to prevent overfitting
5. **Validate**: Use cross-validation for robust evaluation

---

## Next Steps

- [Classification Tutorial](/docs/tutorials/classification/) - For predicting discrete labels
- [Time Series Tutorial](/docs/tutorials/time-series/) - For temporal data
