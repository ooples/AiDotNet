---
title: "NHiTSModel"
description: "Implements N-HiTS (Neural Hierarchical Interpolation for Time Series) for efficient long-horizon forecasting."
section: "Reference"
---

_Time-Series Models_

Implements N-HiTS (Neural Hierarchical Interpolation for Time Series) for efficient long-horizon forecasting.

## For Beginners

N-HiTS improves upon N-BEATS by using a "zoom lens" approach to time series. It looks at your data at three different zoom levels: - Zoomed out (low resolution): Captures long-term trends like yearly seasonality - Medium zoom: Captures medium-term patterns like monthly cycles - Zoomed in (high resolution): Captures short-term fluctuations like daily variations By combining insights from all three levels, it produces more accurate forecasts, especially for predicting far into the future.

## How It Works

N-HiTS is an evolution of N-BEATS that addresses limitations in long-horizon forecasting through: 

Multi-rate data sampling via hierarchical interpolationStack-specific input pooling to capture patterns at different frequenciesMore efficient parameterization compared to N-BEATSInterpolation-based basis functions for smoother predictions

Original paper: Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023). 

**Production-Ready Features:**
- Uses Tensor<T> for GPU-accelerated operations via IEngine
- Proper backpropagation via automatic differentiation
- Vectorized operations - no scalar loops in hot paths
- All parameters are trained (not subsets)

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
    .ConfigureModel(new NHiTSModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"NHiTSModel: forecast {forecast.Length} steps.");
```

