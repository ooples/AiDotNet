---
title: "DynamicRegressionWithARIMAErrors"
description: "Implements a Dynamic Regression model with ARIMA errors for time series forecasting."
section: "Reference"
---

_Time-Series Models_

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

