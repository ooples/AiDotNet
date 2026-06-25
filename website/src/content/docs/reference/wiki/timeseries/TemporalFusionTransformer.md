---
title: "TemporalFusionTransformer<T>"
description: "Implements the Temporal Fusion Transformer (TFT) per Lim et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements the Temporal Fusion Transformer (TFT) per Lim et al. (2021).
Architecture: Input embedding → VSN → LSTM encoder-decoder → Static enrichment →
Interpretable multi-head attention → Gated skip connections → Quantile output.

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
    .ConfigureModel(new TemporalFusionTransformer<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"TemporalFusionTransformer: forecast {forecast.Length} steps.");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CollectAllTrainableParameters` | Collects all trainable parameter tensors from LSTM, GRN, attention, and output layers. |
| `ForwardQuantiles(Tensor<>)` | Forward pass through the full TFT architecture per Lim et al. |

