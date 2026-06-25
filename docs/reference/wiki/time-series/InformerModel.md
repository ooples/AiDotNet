---
title: "InformerModel"
description: "Implements the Informer model for efficient long-sequence time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements the Informer model for efficient long-sequence time series forecasting.

## For Beginners

Informer makes transformers practical for long time series
forecasting. Regular transformers get very slow with long sequences because every time
step looks at every other time step. Informer speeds this up by only looking at the most
important connections (ProbSparse attention), compressing the sequence as it goes through
layers, and predicting all future values at once instead of one at a time.

## How It Works

**The Long-Sequence Forecasting Problem:**
Traditional Transformer models achieve state-of-the-art results in many sequence modeling tasks,
but they struggle with long time series because self-attention has O(L^2) time and memory complexity.
For a sequence of 1000 time steps, vanilla attention requires 1 million operations per layer.
This makes long-horizon forecasting computationally prohibitive.

**The Informer Solution:**
Informer (Zhou et al., AAAI 2021) introduces three key innovations:

1. ProbSparse Self-Attention (O(L log L) complexity)
2. Self-Attention Distilling for sequence compression
3. Generative-Style Decoder for parallel multi-step forecasting

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
    .ConfigureModel(new InformerModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"InformerModel: forecast {forecast.Length} steps.");
```

