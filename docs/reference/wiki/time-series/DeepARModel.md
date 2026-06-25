---
title: "DeepARModel"
description: "Implements DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks."
section: "Reference"
---

_Time-Series Models_

Implements DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.

## For Beginners

DeepAR is like a weather forecaster that doesn't just say "it will be 70 degrees tomorrow" but rather "there's a 50% chance it'll be between 65-75 degrees, a 90% chance it'll be between 60-80 degrees," etc. It uses a type of neural network called LSTM (Long Short-Term Memory) that's good at remembering patterns over time. The "autoregressive" part means it uses its own predictions to make future predictions - similar to how you might predict tomorrow's weather based on today's forecast. This is particularly useful when you need to: - Make decisions based on uncertainty (e.g., inventory planning) - Forecast many related series efficiently (e.g., sales across stores) - Handle new products or stores with limited data

## How It Works

DeepAR is a probabilistic forecasting model that produces full probability distributions rather than point estimates. Key features include: 

Autoregressive RNN architecture (typically LSTM-based)Probabilistic forecasts with quantile predictionsHandles multiple related time seriesBuilt-in handling of covariates and categorical featuresEffective for cold-start scenarios

Original paper: Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020). 

**Production-Ready Features:**
- Uses Tensor<T> for GPU-accelerated operations via IEngine
- Proper LSTM with all gates (input, forget, output, cell)
- Backpropagation through time (BPTT) for gradient computation
- Vectorized operations - no numerical differentiation
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
    .ConfigureModel(new DeepARModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"DeepARModel: forecast {forecast.Length} steps.");
```

