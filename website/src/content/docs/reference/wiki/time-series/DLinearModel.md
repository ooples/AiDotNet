---
title: "DLinearModel"
description: "DLinear — decomposition-linear forecaster (Zeng et al., AAAI 2023, \"Are Transformers Effective for Time Series Forecasting?\")."
section: "Reference"
---

_Time-Series Models_

DLinear — decomposition-linear forecaster (Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series Forecasting?"). The input window is split into a trend (moving average) and a seasonal remainder; a separate linear map projects each to the forecast, and the two are summed. It is deliberately simple yet a strong, current baseline that often matches or beats heavier transformers on long-horizon benchmarks — the right "do we even need attention?" control in any SOTA panel.

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
    .ConfigureModel(new DLinearModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"DLinearModel: forecast {forecast.Length} steps.");
```

