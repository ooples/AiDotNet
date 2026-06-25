---
title: "TimeSeriesRegression<T>"
description: "Represents a time series regression model that incorporates temporal dependencies, trends, and seasonality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Represents a time series regression model that incorporates temporal dependencies, trends, and seasonality.

## For Beginners

This class helps predict future values based on patterns in time-based data.

Think of it like weather forecasting:

- It looks at past weather patterns to predict future weather
- It can recognize long-term trends (like gradual warming)
- It can detect seasonal patterns (like winter being colder than summer)
- It accounts for how recent weather affects tomorrow's weather

This is useful for any data that changes over time, such as stock prices, website traffic,
energy consumption, or sales figures.

## How It Works

The TimeSeriesRegression class extends basic regression by accounting for the temporal structure of the data.
It can model autoregressive components (past values affecting future values), trend components (long-term
directional movement), and seasonal components (recurring patterns at fixed intervals).

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new TimeSeriesRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained TimeSeriesRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesRegression` | Initializes a new instance of the TimeSeriesRegression class with specified options and optional regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Trains the time series regression model on the provided data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAutocorrelationCorrection(Matrix<>,Vector<>)` | Applies autocorrelation correction to improve the model's handling of time-dependent error patterns. |
| `ApplyRegularization` | Applies regularization to the model coefficients. |
| `CalculateAutocorrelation(Vector<>)` | Calculates the first-order autocorrelation coefficient for the given residuals. |
| `CreateNewInstance` | Creates a new instance of the time series regression model with the same configuration. |
| `Deserialize(Byte[])` | Restores the model state from a byte array previously created by the Serialize method. |
| `ExtractCoefficients` | Extracts the relevant coefficients from the model. |
| `ExtractSeasonalCoefficients` | Extracts the seasonal coefficients from the model if seasonality was included in the options. |
| `ExtractTrendCoefficients` | Extracts the trend coefficients from the model if trend was included in the options. |
| `GetOptions` |  |
| `Predict(Matrix<>)` | Predicts target values for the given input features. |
| `PrepareInputData(Matrix<>,Vector<>)` | Prepares the input data by adding lagged features, trend, and seasonal components. |
| `PrepareTargetData(Vector<>)` | Prepares the target data by adjusting for the lag order. |
| `Serialize` | Returns the type of this regression model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | The options that configure this time series regression model. |
| `_regularization` | The regularization strategy used to prevent overfitting. |
| `_timeSeriesModel` | The underlying time series model that handles the core prediction logic. |

