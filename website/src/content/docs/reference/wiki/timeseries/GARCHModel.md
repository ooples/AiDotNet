---
title: "GARCHModel<T>"
description: "Represents a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for time series with changing volatility."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Represents a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for time series with changing volatility.

## For Beginners

GARCH models help predict both the value and the uncertainty of financial data.

Think of it like weather forecasting:

- Regular forecasting predicts the temperature (the mean value)
- GARCH also predicts how much the temperature might vary (the volatility)

For example, with stock prices:

- Some days, prices barely change (low volatility)
- Other days, prices swing wildly up and down (high volatility)
- Often, volatile periods tend to cluster together (volatility clustering)

GARCH helps model this behavior by:

- Using one model to predict the average price (mean model)
- Using another model to predict how much prices might fluctuate (volatility model)

This is especially useful for financial risk management, option pricing, and trading strategies.

## How It Works

The GARCH model is specifically designed for time series data that exhibits volatility clustering, where periods of high 
volatility are followed by periods of high volatility, and periods of low volatility are followed by periods of low volatility.
It combines a mean model (typically an ARIMA model) with a variance model that captures the conditional heteroskedasticity.

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
    .ConfigureModel(new GARCHModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"GARCHModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GARCHModel(GARCHModelOptions<>)` | Initializes a new instance of the `GARCHModel` class with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConditionalVariances(Vector<>)` | Calculates the conditional variances for the time series given the current model parameters. |
| `CalculateGradient(Vector<>,GradientType)` | Calculates the gradient of the log-likelihood function with respect to the model parameters. |
| `CalculateLogLikelihood(Vector<>)` | Calculates the log-likelihood of the data given the current model parameters. |
| `CalculateResidualsAndVariances(Vector<>)` | Calculates the final residuals and conditional variances based on the trained model parameters. |
| `CalculateUnconditionalVariance(Vector<>)` | Calculates the unconditional variance of the time series data. |
| `ConstrainParameters` | Constrains the GARCH parameters to ensure they remain within valid ranges. |
| `CreateInstance` | Creates a new instance of the GARCH model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EstimateParameters(Vector<>)` | Estimates the optimal GARCH parameters using a gradient-based optimization approach. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `GenerateStandardNormal` | Generates a random number from a standard normal distribution. |
| `GetModelMetadata` | Returns metadata about the model, including its type, parameters, and configuration. |
| `InitializeParameters` | Initializes the GARCH model parameters with reasonable starting values. |
| `Predict(Matrix<>)` | Generates predictions for the given input data, including both mean and volatility forecasts. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input vector. |
| `Reset` | Resets the model to its initial state. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the GARCH model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The coefficients for the ARCH terms in the variance equation. |
| `_beta` | The coefficients for the GARCH terms in the variance equation. |
| `_conditionalVariances` | The estimated conditional variances for each time point in the series. |
| `_garchOptions` | The configuration options for the GARCH model. |
| `_meanModel` | The model used to forecast the mean (average value) of the time series. |
| `_omega` | The constant term in the GARCH variance equation. |
| `_residuals` | The residuals (errors) from the mean model's predictions. |

