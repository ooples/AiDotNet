---
title: "VectorAutoRegressionModel<T>"
description: "Implements a Vector Autoregression (VAR) model for multivariate time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements a Vector Autoregression (VAR) model for multivariate time series forecasting.

## For Beginners

A VAR model helps you forecast multiple related time series at once, accounting for how they
influence each other.

For example, if you're analyzing economic data, a VAR model could simultaneously forecast:

- GDP growth
- Unemployment rate
- Inflation rate

While accounting for relationships like:

- How unemployment affects future GDP
- How GDP affects future inflation
- How each variable's past values affect its own future

Think of it as a system that recognizes the interconnected nature of multiple time series
and uses these connections to make better forecasts for all variables simultaneously.

## How It Works

The Vector Autoregression (VAR) model is a multivariate extension of the univariate autoregressive model.
It captures linear dependencies among multiple time series variables, where each variable is modeled as
a function of past values of itself and past values of other variables in the system.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;

double[] series =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};
var x = new Matrix<double>(series.Length, 1);
for (int i = 0; i < series.Length; i++) x[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new VectorAutoRegressionModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"VectorAutoRegressionModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VectorAutoRegressionModel` | Initializes a new instance with default settings. |
| `VectorAutoRegressionModel(VARModelOptions<>)` | Initializes a new instance of the VectorAutoRegressionModel class with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the coefficient matrix of the VAR model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateResiduals(Matrix<>)` | Calculates the residuals (errors) between the model's predictions and the actual values. |
| `CholeskyDecomposition(Matrix<>)` | Performs Cholesky decomposition of a symmetric positive definite matrix. |
| `ConstructVARMatrix` | Constructs the VAR coefficient matrix in companion form. |
| `CreateInstance` | Creates a new instance of the VectorAutoRegressionModel class. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EstimateOLS(Matrix<>,Vector<>)` | Estimates coefficients using Ordinary Least Squares (OLS) regression. |
| `EstimateResidualCovariance` | Estimates the covariance matrix of residuals. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the performance of the trained model on test data. |
| `Forecast(Matrix<>,Int32)` | Generates multi-step forecasts for all variables in the VAR system. |
| `GetModelMetadata` | Gets metadata about the VAR model. |
| `ImpulseResponseAnalysis(Int32)` | Analyzes the dynamic relationships between variables in the VAR system. |
| `IsValidValue()` | Checks if a value is valid (not NaN or infinity). |
| `Predict(Matrix<>)` | Generates forecasts using the trained VAR model. |
| `PredictSingle(Vector<>)` | Predicts a single value from one of the variables in the VAR system. |
| `PredictSingleStep(Matrix<>)` | Produces a single-step multivariate prediction from a lag window. |
| `PrepareLaggedData(Matrix<>)` | Prepares a matrix of lagged data for VAR model estimation. |
| `Reset` | Resets the VAR model to its untrained state. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the model-specific training logic for the VAR model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | Matrix of coefficients that capture the relationships between variables across time lags. |
| `_intercepts` | Vector of intercept terms for each equation in the VAR model. |
| `_residuals` | Matrix of residuals (errors) from the model fit. |
| `_trainingSeries` | Stored training series for in-sample predictions. |
| `_varOptions` | Configuration options specific to the VAR model. |

