---
title: "AutoformerModel"
description: "Implements the Autoformer model for long-term time series forecasting with decomposition."
section: "Reference"
---

_Time-Series Models_

Implements the Autoformer model for long-term time series forecasting with decomposition.

## For Beginners

Autoformer is like having two experts work together:

- One expert focuses on the long-term direction (trend)
- One expert focuses on repeating patterns (seasonality)

Instead of looking at individual data points, it looks at how patterns repeat over time.
If today's pattern looks like last week's pattern, that's useful information!

Example use cases:

- Electricity demand forecasting (daily/weekly patterns)
- Retail sales prediction (seasonal buying patterns)
- Traffic flow prediction (rush hour patterns)

## How It Works

**The Long-Term Forecasting Challenge:**
Long-term time series forecasting requires models that can capture both fine-grained seasonal
patterns and long-term trends. Traditional approaches struggle because:

- RNNs have difficulty with long-range dependencies
- Transformers treat time series like text, ignoring continuous nature
- Neither explicitly models trend and seasonality separately

**The Autoformer Solution (Wu et al., NeurIPS 2021):**
Autoformer introduces three key innovations:

1. **Series Decomposition Block:**

Progressive separation of trend and seasonal components at each layer.
Uses moving average to extract trend, remainder is seasonal.
Formula: Trend = MovingAvg(X), Seasonal = X - Trend

2. **Auto-Correlation Mechanism:**

Replaces point-wise self-attention with period-based dependencies.
Uses FFT to find correlations between sub-series efficiently (O(L log L)).
Aggregates similar sub-sequences based on their correlation strength.

3. **Progressive Decomposition Architecture:**

Each encoder/decoder layer further refines the decomposition.
Seasonal and trend branches are processed separately and accumulated.

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
    .ConfigureModel(new AutoformerModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"AutoformerModel: forecast {forecast.Length} steps.");
```

