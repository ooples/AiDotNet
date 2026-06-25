---
title: "NBEATSModel<T>"
description: "Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting.

## For Beginners

N-BEATS is a state-of-the-art neural network for time series
forecasting that automatically learns patterns from your data. Unlike traditional methods
that require you to manually specify trends and seasonality, N-BEATS figures these out
on its own.

Key advantages:

- No need for manual feature engineering (the model learns what's important)
- Can capture complex, non-linear patterns
- Provides interpretable components (trend, seasonality) when configured to do so
- Works well for both short-term and long-term forecasting

The model works by stacking many "blocks" together, where each block tries to:

1. Understand what patterns are in the input (backcast)
2. Predict the future based on those patterns (forecast)
3. Pass the unexplained patterns to the next block

This allows the model to decompose complex time series into simpler components.

## How It Works

N-BEATS is a deep neural architecture based on backward and forward residual links and
a very deep stack of fully-connected layers. The architecture has the following key features:

The original paper: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting" (ICLR 2020).

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NBEATSModel(NBEATSModelOptions<>)` | Initializes a new instance of the NBEATSModel class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters in the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateInstance` | Creates a new instance of the N-BEATS model. |
| `CreateSliceWeights(Int32,Int32,INumericOperations<>)` | Creates slice weights for extracting a single element from a vector. |
| `DeserializeCore(BinaryReader)` | Deserializes model-specific data from the binary reader. |
| `ExtractLookbackWindow(Matrix<>,Vector<>,Int32)` | Extracts a lookback window vector for a given sample index. |
| `ExtractNormalizedLookbackWindow(Matrix<>,Vector<>,Int32)` | Extracts a normalized lookback window for training. |
| `ForecastHorizon(Vector<>)` | Generates forecasts for multiple future time steps. |
| `GetModelMetadata` | Gets metadata about the N-BEATS model. |
| `GetParameters` | Gets all model parameters as a single vector. |
| `InitializeBlocks` | Initializes all blocks in the N-BEATS architecture. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the provided input vector. |
| `SerializeCore(BinaryWriter)` | Serializes model-specific data to the binary writer. |
| `SetParameters(Vector<>)` | Sets all model parameters from a single vector. |
| `TrainCore(Matrix<>,Vector<>)` | Trains the N-BEATS model using tape-based automatic differentiation with Adam optimizer. |
| `ValidateNBEATSOptions` | Validates the N-BEATS specific options. |

