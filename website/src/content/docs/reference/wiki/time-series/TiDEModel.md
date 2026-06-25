---
title: "TiDEModel"
description: "TiDE — Time-series Dense Encoder (Das et al., TMLR 2023)."
section: "Reference"
---

_Time-Series Models_

TiDE — Time-series Dense Encoder (Das et al., TMLR 2023). A pure-MLP forecaster: a ReLU encoder maps the input window to a latent, a decoder projects it to the forecast, and a linear residual skips the window straight to the output. Despite using no attention it matches or beats transformers on long-horizon benchmarks at far lower cost — a strong, current member of the SOTA panel. Implemented with explicit forward + manual backprop (a 1-hidden-layer ReLU MLP + linear skip), so every gradient is exact.

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
    .ConfigureModel(new TiDEModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"TiDEModel: forecast {forecast.Length} steps.");
```

