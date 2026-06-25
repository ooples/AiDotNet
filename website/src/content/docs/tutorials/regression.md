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
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// House features: sqft, bedrooms, bathrooms, age
double[][] features =
{
    new[] { 1400.0, 3.0, 2.0, 15.0 }, new[] { 1600.0, 3.0, 2.0, 10.0 },
    new[] { 1700.0, 3.0, 2.5, 5.0 },  new[] { 1875.0, 4.0, 3.0, 8.0 },
    new[] { 1100.0, 2.0, 1.0, 25.0 }, new[] { 2200.0, 4.0, 3.0, 2.0 }
};
double[] prices = { 245000, 312000, 279000, 308000, 199000, 425000 };

var X = ToMatrix(features);
var y = new Vector<double>(prices);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

var newHome = new Matrix<double>(1, 4);
foreach (var (v, j) in new[] { 1500.0, 3.0, 2.0, 12.0 }.Select((v, j) => (v, j)))
    newHome[0, j] = v;
Console.WriteLine($"Predicted price: ${result.Predict(newHome)[0]:N0}");

// Metrics come off the result — no hand-rolled math.
var stats = result.GetDataSetStats(X, y);
Console.WriteLine($"R²: {stats.PredictionStats.R2:F4}, RMSE: {stats.ErrorStats.RMSE:N0}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

---

## Available Regressors

Swap the `ConfigureModel(...)` argument to change algorithm — everything else stays the same.

### Tree-Based Methods

| Regressor | Best For |
|:----------|:---------|
| `RandomForestRegression` | General purpose, robust |
| `GradientBoostingRegression` | High accuracy |
| `DecisionTreeRegression` | Interpretability |

### Linear Methods

| Regressor | Best For |
|:----------|:---------|
| `SimpleRegression` / `MultipleRegression` | Ordinary least squares |
| `RidgeRegression` | Multicollinearity (L2) |
| `LassoRegression` | Feature selection (L1) |
| `ElasticNetRegression` | L1 + L2 |
| `PolynomialRegression` | Non-linear relationships |

### Distance-Based & Neural

| Regressor | Best For |
|:----------|:---------|
| `KNearestNeighborsRegression` | Non-linear, small datasets |
| `NeuralNetworkRegression` | Complex non-linear patterns |

### Neural Network Regression

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1400.0, 3.0 }, new[] { 1600.0, 3.0 }, new[] { 1700.0, 3.0 },
    new[] { 1875.0, 4.0 }, new[] { 1100.0, 2.0 }, new[] { 2200.0, 4.0 }
};
double[] targets = { 245000, 312000, 279000, 308000, 199000, 425000 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new NeuralNetworkRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine($"Trained: {result.TotalTrainableParameters:N0} parameters");
```

---

## Data Preprocessing

Pass a preprocessing pipeline to `ConfigurePreprocessing(...)` to scale or impute features before training — useful for linear and distance-based models. Regularized linear models like `RidgeRegression` pair well with feature scaling.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1400.0, 3.0 }, new[] { 1600.0, 3.0 }, new[] { 1700.0, 3.0 }, new[] { 1875.0, 4.0 }
};
double[] targets = { 245000, 312000, 279000, 308000 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RidgeRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained a regularized linear model.");
```

---

## Evaluation Metrics

Every regression metric is computed for you and lives under `result.GetDataSetStats(X, y)`.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] data =
{
    new[] { 1400.0, 3.0 }, new[] { 1600.0, 3.0 }, new[] { 1700.0, 3.0 },
    new[] { 1875.0, 4.0 }, new[] { 1100.0, 2.0 }, new[] { 2200.0, 4.0 }
};
double[] targets = { 245000, 312000, 279000, 308000, 199000, 425000 };

var X = ToMatrix(data);
var y = new Vector<double>(targets);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GradientBoostingRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

var stats = result.GetDataSetStats(X, y);
Console.WriteLine($"MAE:  {stats.ErrorStats.MAE:F4}");
Console.WriteLine($"MSE:  {stats.ErrorStats.MSE:F4}");
Console.WriteLine($"RMSE: {stats.ErrorStats.RMSE:F4}");
Console.WriteLine($"R²:   {stats.PredictionStats.R2:F4}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

---

## Best Practices

1. **Start simple**: Use `MultipleRegression` as a baseline.
2. **Check for outliers**: Outliers heavily influence linear models.
3. **Feature scaling**: Most algorithms benefit from `ConfigurePreprocessing()`.
4. **Regularize**: Use `RidgeRegression` or `LassoRegression` to prevent overfitting.
5. **Validate**: Add `ConfigureCrossValidation(...)` for robust evaluation.

---

## Next Steps

- [Classification Tutorial](/docs/tutorials/classification/) - For predicting discrete labels
- [Time Series Tutorial](/docs/tutorials/time-series/) - For temporal data
