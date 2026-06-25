---
title: "MAModel<T>"
description: "Implements a Moving Average (MA) model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements a Moving Average (MA) model for time series forecasting.

## For Beginners

A Moving Average (MA) model predicts future values based on past prediction errors.

Think of it like this: If you've been consistently underestimating or overestimating 
values in the past, the MA model learns from these mistakes and adjusts future predictions.

For example, if a weather forecast has been consistently underestimating temperatures 
by 2 degrees for several days, an MA model would learn this pattern and adjust its 
future predictions upward.

The key parameter of an MA model is 'q', which determines how many past prediction 
errors to consider. For instance, with q=3, the model looks at errors from the last 
three periods when making a new prediction.

## How It Works

MA models predict future values based on past prediction errors (residuals). 
The model is defined as: Yt = μ + et + ?1et-1 + ?2et-2 + ... + ?qet-q
where Yt is the value at time t, μ is the mean, et is the error term at time t,
and ?i are the MA coefficients.

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
    .ConfigureModel(new MAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"MAModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAModel(MAModelOptions<>)` | Creates a new MA model with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateGradient(Vector<>,Vector<>,Vector<>)` | Calculates the gradient of the negative log-likelihood with respect to the MA coefficients. |
| `CalculateNegativeLogLikelihood(Vector<>,Vector<>)` | Calculates the negative log-likelihood of the MA model given the parameters. |
| `CalculateRecentErrors(Vector<>)` | Calculates the most recent errors for making future predictions. |
| `CalculateSearchDirection(Matrix<>,Vector<>)` | Calculates the search direction for optimization using the approximated Hessian. |
| `Clone` | Flag indicating whether the model has been trained. |
| `CreateInstance` | Creates a new instance of the MA model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's state from a binary stream. |
| `EstimateMACoefficients(Vector<>,Int32)` | Estimates the Moving Average coefficients using maximum likelihood estimation. |
| `EstimateNoiseVariance(Vector<>,Vector<>)` | Estimates the variance of the white noise process in the MA model. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `GetModelMetadata` | Gets metadata about the model, including its type, parameters, and configuration. |
| `InitialMACoefficientsEstimate(Vector<>,Int32)` | Provides initial estimates of MA coefficients based on autocorrelation. |
| `LineSearch(Vector<>,Vector<>,Vector<>,)` | Performs line search to find an appropriate step size for the optimization algorithm. |
| `OptimizeMACoefficients(Vector<>,Vector<>)` | Optimizes MA coefficients using numerical optimization with the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm. |
| `Predict(Matrix<>)` | Makes predictions using the trained MA model. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input vector. |
| `SerializeCore(BinaryWriter)` | Serializes the model's state to a binary stream. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the MA model. |
| `UpdateHessianApproximation(Vector<>,Matrix<>,Vector<>,Vector<>,Vector<>)` | Updates the approximation of the Hessian matrix using the BFGS update formula. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_convergenceTolerance` | Convergence tolerance for optimization algorithms. |
| `_maCoefficients` | Coefficients for the moving average component of the model. |
| `_maOptions` | Options specific to the MA model, including the order (q) parameter. |
| `_maxIterations` | Maximum number of iterations for optimization algorithms. |
| `_mean` | The mean of the time series, used as a baseline for predictions. |
| `_noiseVariance` | The variance of the white noise process. |
| `_recentErrors` | The most recent errors (residuals) from the model's predictions. |

