---
title: "SARIMAModel<T>"
description: "Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting.

## For Beginners

SARIMA is used to predict future values in a time series (data collected over time) that has seasonal patterns.
Think of it like predicting ice cream sales throughout the year - there's a general trend (maybe increasing over years)
and a seasonal pattern (higher in summer, lower in winter). SARIMA can capture both these patterns.

The model has several parameters:

- p: How many previous values influence the current value
- d: How many times we need to subtract consecutive values to make the data stable
- q: How many previous prediction errors influence the current prediction
- P, D, Q: The same as above, but for seasonal patterns
- m: The length of the seasonal cycle (e.g., 12 for monthly data with yearly patterns)

## How It Works

The SARIMA model extends the ARIMA model by incorporating seasonal components, making it suitable for 
data with seasonal patterns. It combines autoregressive (AR), integrated (I), and moving average (MA) 
components for both seasonal and non-seasonal parts of the time series.

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
    .ConfigureModel(new SARIMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"SARIMAModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SARIMAModel` | Initializes a new instance with default settings. |
| `SARIMAModel(SARIMAOptions<>)` | Initializes a new instance of the SARIMAModel class with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDifferencing(Vector<>)` | Applies both seasonal and non-seasonal differencing to the input series. |
| `CalculateARSARResiduals(Vector<>)` | Calculates residuals after applying AR and seasonal AR components. |
| `CreateInstance` | Creates a new instance of the SARIMA model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EstimateConstant(Vector<>)` | Estimates the constant term for the SARIMA model. |
| `EstimateSeasonalARCoefficients(Vector<>)` | Estimates the seasonal autoregressive (SAR) coefficients. |
| `EstimateSeasonalMACoefficients(Vector<>)` | Estimates the seasonal moving average (SMA) coefficients. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the performance of the trained model on test data. |
| `Forecast(Vector<>,Int32)` | Forecasts future values using the trained SARIMA model, properly handling both regular and seasonal differencing. |
| `GetARParameters` | Gets the non-seasonal autoregressive (AR) parameters of the model. |
| `GetMAParameters` | Gets the non-seasonal moving average (MA) parameters of the model. |
| `GetModelMetadata` | Gets metadata about the model, including its type, parameters, and configuration. |
| `GetSeasonalARParameters` | Gets the seasonal autoregressive (SAR) parameters of the model. |
| `GetSeasonalMAParameters` | Gets the seasonal moving average (SMA) parameters of the model. |
| `GetSeasonalPeriod` | Gets the seasonal period used in the model. |
| `Predict(Matrix<>)` | Generates predictions using the trained SARIMA model. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input vector. |
| `SeasonalDifference(Vector<>,Int32)` | Applies seasonal differencing to the input series. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the SARIMA model. |
| `UndoRegularDifferencing(List<>,Vector<>)` | Undoes regular (non-seasonal) differencing by computing tail values at each integration level and cumulatively summing in reverse order. |
| `UndoSeasonalDifferencing(List<>,Vector<>)` | Undoes seasonal differencing D times by reconstructing original-scale values from the seasonal tail of the history at each level. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_D` | Order of the seasonal differencing. |
| `_P` | Order of the seasonal autoregressive component. |
| `_Q` | Order of the seasonal moving average component. |
| `_arCoefficients` | Coefficients for the non-seasonal autoregressive (AR) component. |
| `_constant` | The constant term in the SARIMA model. |
| `_d` | Order of the non-seasonal differencing. |
| `_lastTrainDiffValues` | Stored differenced training values for initializing Predict(Matrix) state. |
| `_lastTrainResiduals` | Stored AR+SAR residuals from training for initializing Predict(Matrix) state. |
| `_m` | The seasonal period (e.g., 12 for monthly data with yearly seasonality). |
| `_maCoefficients` | Coefficients for the non-seasonal moving average (MA) component. |
| `_p` | Order of the non-seasonal autoregressive component. |
| `_q` | Order of the non-seasonal moving average component. |
| `_sarCoefficients` | Coefficients for the seasonal autoregressive (SAR) component. |
| `_sarimaOptions` | Stores the configuration options for the SARIMA model. |
| `_smaCoefficients` | Coefficients for the seasonal moving average (SMA) component. |
| `_trainingSeries` | Original training series for in-sample Predict(Matrix) with undifferencing. |

