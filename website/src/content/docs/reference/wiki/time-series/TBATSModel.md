---
title: "TBATSModel"
description: "Implements the TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components) model for complex time series forecasting with multiple seasonal patterns."
section: "Reference"
---

_Time-Series Models_

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

