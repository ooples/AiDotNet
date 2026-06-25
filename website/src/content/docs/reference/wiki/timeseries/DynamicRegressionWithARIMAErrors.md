---
title: "DynamicRegressionWithARIMAErrors<T>"
description: "Implements a Dynamic Regression model with ARIMA errors for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements a Dynamic Regression model with ARIMA errors for time series forecasting.

## How It Works

This model combines regression analysis with ARIMA (AutoRegressive Integrated Moving Average) error modeling.
It first models the relationship between the target variable and external predictors using regression,
then applies ARIMA modeling to the residuals to capture temporal patterns in the error terms.

For Beginners:
Dynamic Regression with ARIMA Errors is like having two powerful forecasting tools working together:

1. Regression Component: This part captures how external factors (like temperature, price changes,

or marketing campaigns) affect what you're trying to predict. For example, if you're forecasting
ice cream sales, this component would measure how much each degree of temperature increases sales.

2. ARIMA Error Component: After accounting for external factors, there are often still patterns

in the data that the regression alone can't explain. The ARIMA component captures these patterns
by looking at:

- Past values (AR - AutoRegressive)
- Trends removed through differencing (I - Integrated)
- Past prediction errors (MA - Moving Average)

When combined, these components create a powerful forecasting model that can:

- Account for the impact of known external factors
- Capture complex temporal patterns in the data
- Handle both stationary and non-stationary time series
- Provide more accurate forecasts than either approach alone

This model is particularly useful when you have both:

- External variables that influence your target variable
- Temporal patterns that persist in the data after accounting for these external influences

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
    .ConfigureModel(new DynamicRegressionWithARIMAErrors<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"DynamicRegressionWithARIMAErrors: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynamicRegressionWithARIMAErrors` | Initializes a new instance with default settings. |
| `DynamicRegressionWithARIMAErrors(DynamicRegressionWithARIMAErrorsOptions<>)` | Creates a new Dynamic Regression with ARIMA Errors model with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyRegularization` | Applies regularization to the regression coefficients. |
| `CalculateAutocorrelations(Vector<>,Int32)` | Calculates autocorrelations of a time series up to a specified lag. |
| `Clone` | Creates a deep copy of the current model. |
| `ComputeARGradient(Vector<>,Int32)` | Computes the gradient for an AR coefficient. |
| `ComputeLogLikelihood(Vector<>)` | Computes the log-likelihood of the current ARMA model. |
| `ComputeMAGradient(Vector<>,Int32)` | Computes the gradient for an MA coefficient. |
| `ComputeModelResiduals(Vector<>)` | Computes residuals using the current ARMA model. |
| `CreateInstance` | Creates a new instance of the model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's state from a binary stream. |
| `DifferenceTimeSeries(Vector<>,Int32)` | Applies differencing to the time series to make it stationary. |
| `EnsureARStationarity` | Ensures the AR process is stationary by scaling coefficients if necessary. |
| `EnsureMAInvertibility` | Ensures the MA process is invertible by scaling coefficients if necessary. |
| `EstimateARCoefficients([],Int32)` | Estimates the AR coefficients using the Yule-Walker equations. |
| `EstimateMACoefficients(Vector<>,[],Int32)` | Estimates the MA coefficients using the innovation algorithm. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `ExtractResiduals(Matrix<>,Vector<>)` | Calculates the residuals from the regression model. |
| `FitARIMAModel(Vector<>)` | Fits an ARIMA model to the residuals from the regression component. |
| `FitRegressionModel(Matrix<>,Vector<>)` | Fits the regression component of the model to the data. |
| `Forecast(Vector<>,Int32,Matrix<>)` | Forecasts future values based on a history of time series data and exogenous variables. |
| `GetModelMetadata` | Gets metadata about the trained model. |
| `InverseDifferenceTimeSeries(Vector<>,Vector<>)` | Reverses the differencing process to convert predictions back to the original scale. |
| `NormalizeRegressionCoefficients` | Normalizes the regression coefficients to have unit norm. |
| `OptimizeARMACoefficients(Vector<>)` | Optimizes the AR and MA coefficients jointly to improve model fit. |
| `Predict(Matrix<>)` | Makes predictions using the trained model. |
| `PredictSingle(Vector<>)` | Predicts a single value based on a single input vector of external regressors. |
| `Reset` | Resets the model to its untrained state. |
| `SerializeCore(BinaryWriter)` | Serializes the model's state to a binary stream. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the core training algorithm for the Dynamic Regression with ARIMA Errors model. |
| `UpdateARCoefficients(Vector<>)` | Updates the AR coefficients using gradient descent. |
| `UpdateIntercept` | Updates the intercept based on differencing. |
| `UpdateMACoefficients(Vector<>)` | Updates the MA coefficients using gradient descent. |
| `UpdateModelParameters` | Updates and optimizes model parameters before making predictions. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_arCoefficients` | Coefficients for the autoregressive (AR) component of the model. |
| `_arimaOptions` | Configuration options for the Dynamic Regression with ARIMA Errors model. |
| `_differenced` | Values needed to reverse differencing when making predictions. |
| `_intercept` | The constant term (intercept) in the regression equation. |
| `_maCoefficients` | Coefficients for the moving average (MA) component of the model. |
| `_regressionCoefficients` | Coefficients for the regression component, representing the impact of external variables. |
| `_regularization` | Regularization method to prevent overfitting in the regression component. |
| `_trainingSeries` | Stored training series for in-sample predictions. |

