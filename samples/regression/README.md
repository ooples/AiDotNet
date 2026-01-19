# Regression Samples

This directory contains examples of regression algorithms in AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [PricePrediction](./PricePrediction/) | Predict house prices using ensemble methods |
| [DemandForecasting](./DemandForecasting/) | Forecast product demand |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Regression;

var features = new double[][] { /* features */ };
var targets = new double[] { /* continuous values */ };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingRegression<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, targets);

// Use result.Predict() directly - this is the facade pattern
var prediction = result.Predict(newFeatures);
```

## Available Regressors

AiDotNet includes 41+ regression algorithms:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Polynomial Regression
- Gradient Boosting Regression
- Random Forest Regression
- Support Vector Regression
- Gaussian Process Regression
- Neural Network regressors

## Learn More

- [Regression Tutorial](/docs/tutorials/regression/)
- [API Reference](/api/AiDotNet.Regression/)
