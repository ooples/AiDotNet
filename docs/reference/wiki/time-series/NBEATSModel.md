---
title: "NBEATSModel"
description: "Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting."
section: "Reference"
---

_Time-Series Models_

Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting.

## For Beginners

N-BEATS is a state-of-the-art neural network for time series forecasting that automatically learns patterns from your data. Unlike traditional methods that require you to manually specify trends and seasonality, N-BEATS figures these out on its own. Key advantages: - No need for manual feature engineering (the model learns what's important) - Can capture complex, non-linear patterns - Provides interpretable components (trend, seasonality) when configured to do so - Works well for both short-term and long-term forecasting The model works by stacking many "blocks" together, where each block tries to: 1. Understand what patterns are in the input (backcast) 2. Predict the future based on those patterns (forecast) 3. Pass the unexplained patterns to the next block This allows the model to decompose complex time series into simpler components.

## How It Works

N-BEATS is a deep neural architecture based on backward and forward residual links and a very deep stack of fully-connected layers. The architecture has the following key features: 

Doubly residual stacking: Each block produces a backcast (reconstruction) and forecastHierarchical decomposition: Multiple stacks focus on different aspects (trend, seasonality)Interpretability: Can use polynomial and Fourier basis for explainable forecastsNo manual feature engineering: Learns directly from raw time series data

The original paper: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting" (ICLR 2020).

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
    .ConfigureModel(new NBEATSModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"NBEATSModel: forecast {forecast.Length} steps.");
```

