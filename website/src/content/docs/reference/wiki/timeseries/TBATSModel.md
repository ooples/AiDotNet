---
title: "TBATSModel<T>"
description: "Implements the TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components) model for complex time series forecasting with multiple seasonal patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements the TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components) model
for complex time series forecasting with multiple seasonal patterns.

## For Beginners

TBATS is like a Swiss Army knife for time series forecasting. It can handle complex data with:

- Multiple seasonal patterns (e.g., daily, weekly, and yearly patterns all at once)
- Non-linear growth (using Box-Cox transformations)
- Autocorrelated errors (using ARMA models)

For example, if you're analyzing hourly electricity demand, TBATS can simultaneously model:

- Daily patterns (people use more electricity during the day than at night)
- Weekly patterns (usage differs on weekdays versus weekends)
- Yearly patterns (more electricity is used for heating in winter or cooling in summer)

This makes TBATS particularly useful for complex forecasting problems where simpler methods fail.

## How It Works

The TBATS model is an advanced exponential smoothing method that can handle multiple seasonal patterns
of different lengths. It uses trigonometric functions to model seasonality, Box-Cox transformations
to handle non-linearity, and ARMA processes to model residual correlations.

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
    .ConfigureModel(new TBATSModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"TBATSModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TBATSModel(TBATSModelOptions<>)` | Initializes a new instance of the TBATSModel class with optional configuration options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAutocorrelations(Vector<>,Int32)` | Calculates autocorrelations of the time series data. |
| `CalculateAverageSeasonalFactor(Vector<>,Int32)` | Calculates the average seasonal factor for JIT compilation approximation. |
| `CalculateLogLikelihood(Vector<>)` | Calculates the log-likelihood of the model given the observed data. |
| `CalculateResiduals(Vector<>)` | Calculates residuals between the observed values and the model's predictions. |
| `CalculateRobustAutocorrelations(Vector<>,Int32)` | Calculates robust autocorrelations that are less sensitive to outliers. |
| `CalculateRobustResiduals(Vector<>)` | Calculates robust residuals using Huber's M-estimator. |
| `CalculateTypicalARMAContribution` | Calculates a typical ARMA contribution for JIT approximation. |
| `CreateInstance` | Creates a new instance of the TBATS model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `DurbinLevinsonAlgorithm([],Int32)` | Implements the Durbin-Levinson algorithm for estimating AR coefficients. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `GetModelMetadata` | Gets metadata about the model, including its type, configuration, and learned parameters. |
| `InitializeARMACoefficients(Vector<>)` | Initializes ARMA coefficients using standard (non-robust) methods. |
| `InitializeARMACoefficientsRobust(Vector<>)` | Initializes ARMA coefficients using robust methods. |
| `InitializeComponents(Vector<>)` | Initializes all components of the TBATS model using robust methods. |
| `InitializeSeasonalComponent(Vector<>,Int32)` | Initializes a seasonal component using standard (non-robust) methods. |
| `InitializeSeasonalComponentRobust(Vector<>,Int32)` | Initializes a seasonal component using robust methods. |
| `InnovationsAlgorithm([],Int32)` | Implements the innovations algorithm for estimating MA coefficients. |
| `Predict(Matrix<>)` | Generates forecasts using the trained TBATS model. |
| `PredictSingle(Vector<>)` | Predicts a single value for the given input vector. |
| `Reset` | Resets the model to its initial state. |
| `RobustLinearRegression(Vector<>,Vector<>)` | Performs robust linear regression using the Theil-Sen estimator. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Performs the core training logic for the TBATS model. |
| `UpdateARMACoefficients(Vector<>)` | Updates the ARMA coefficients based on the observed data. |
| `UpdateComponents(Vector<>)` | Updates the level, trend, and seasonal components based on the observed data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_arCoefficients` | The autoregressive (AR) coefficients for the ARMA error model. |
| `_boxCoxLambda` | The Box-Cox transformation parameter for handling non-linearity. |
| `_level` | The level component of the time series, representing the current base value. |
| `_maCoefficients` | The moving average (MA) coefficients for the ARMA error model. |
| `_seasonalComponents` | The seasonal components of the time series, one for each seasonal period. |
| `_tbatsOptions` | Configuration options for the TBATS model. |
| `_trainingSeries` | Stored training series for in-sample predictions. |
| `_trend` | The trend component of the time series, representing the rate of change. |

