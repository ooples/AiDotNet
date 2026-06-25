---
title: "NLinearModel"
description: "NLinear — normalization-linear forecaster (Zeng et al., AAAI 2023)."
section: "Reference"
---

_Time-Series Models_

NLinear — normalization-linear forecaster (Zeng et al., AAAI 2023). Subtracts the last value of the input window (a simple per-window normalization that absorbs level/distribution shift), applies one linear map, then adds the last value back. With DLinear it forms the pair of strong, current linear baselines that frequently rival transformers on long-horizon forecasting.

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
    .ConfigureModel(new NLinearModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"NLinearModel: forecast {forecast.Length} steps.");
```

