---
title: "STLDecomposition<T>"
description: "Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis.

## For Beginners

STL decomposition is like breaking down a song into its basic elements - the melody (trend),
the repeating chorus (seasonal pattern), and the unique variations (residuals).

For example, if you analyze monthly sales data:

- The trend component shows the long-term increase or decrease in sales
- The seasonal component shows regular patterns that repeat (like higher sales during holidays)
- The residual component shows what's left after removing trend and seasonality (like unexpected events)

This decomposition helps you understand what's driving your time series and can improve forecasting.
The model offers different algorithms (standard, robust, and fast) to handle various types of data.

## How It Works

STL decomposition breaks down a time series into three components: trend, seasonal, and residual.
It uses locally weighted regression (LOESS) to extract these components, making it robust to
outliers and applicable to a wide range of time series data.

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
    .ConfigureModel(new STLDecomposition<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"STLDecomposition: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STLDecomposition(STLDecompositionOptions<>)` | Initializes a new instance of the STLDecomposition class with optional configuration options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeSeasonalEvolution` | Analyzes how the seasonal pattern evolves over time. |
| `ApplyRobustnessWeights(Vector<>,Vector<>)` | Applies robustness weights to the input data. |
| `CalculateResiduals(Vector<>,Vector<>,Vector<>)` | Calculates the residual component by subtracting trend and seasonal components from the original data. |
| `CalculateRobustWeights(Vector<>)` | Calculates robustness weights based on residuals to reduce the influence of outliers. |
| `CalculateSeasonalAutocorrelation` | Calculates the autocorrelation of the seasonal component at the seasonal lag. |
| `CalculateSeasonalStrength` | Calculates the seasonal strength of the time series. |
| `CalculateTrendStrength` | Calculates the trend strength of the time series. |
| `CreateInstance` | Creates a new instance of the STL decomposition model with the same options. |
| `CycleSubseriesSmoothing(Vector<>,Int32,Int32)` | Performs cycle-subseries smoothing on the input data. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the performance of the trained model on test data. |
| `ExtractLocalSeasonalPattern(Vector<>,Int32,Int32)` | Extracts a seasonal pattern from a local window of the time series. |
| `GetModelMetadata` | Gets metadata about the model, including its type, configuration, and information about the decomposed components. |
| `GetResidual` | Gets the residual component of the time series. |
| `GetSeasonal` | Gets the seasonal component of the time series. |
| `GetTrend` | Gets the trend component of the time series. |
| `HasSignificantSeasonalEvolution(Vector<>)` | Determines if the seasonal pattern is evolving significantly over time. |
| `IsInvalidValue()` | Checks if a value is invalid (NaN or Infinity). |
| `LoessSmoothing(List<ValueTuple<,>>,Int32)` | Performs LOESS smoothing on a list of (x,y) points using a distance window. |
| `LoessSmoothing(Vector<>,Int32)` | Performs LOESS (locally weighted regression) smoothing on the input data. |
| `LowPassFilter(Vector<>,Int32)` | Applies a low-pass filter to the input data. |
| `MovingAverage(Vector<>,Int32)` | Calculates a moving average of the input data with the specified window size. |
| `NormalizeSeasonal` | Normalizes the seasonal component to ensure it sums to zero, adjusting the trend accordingly. |
| `PerformAdaptiveSTL(Vector<>)` | Improves the standard STL algorithm by attempting to detect and adapt to changing seasonality. |
| `PerformEvolvingSeasonalitySTL(Vector<>)` | Performs STL decomposition adapted for time series with evolving seasonality. |
| `PerformFastSTL(Vector<>)` | Performs a faster version of the STL decomposition algorithm. |
| `PerformRobustSTL(Vector<>)` | Performs the robust STL decomposition algorithm, which is less sensitive to outliers. |
| `PerformStandardSTL(Vector<>)` | Performs the standard STL decomposition algorithm. |
| `Predict(Matrix<>)` | Generates forecasts for future time periods based on the decomposed components. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the decomposed components. |
| `Reset` | Resets the model to its initial state. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `SmoothSeasonal(Vector<>,Int32,Int32)` | Smooths the seasonal component by applying a moving average to each subseries. |
| `SmoothSeasonalTransitions(Vector<>,Int32)` | Smooths transitions between adjacent seasonal patterns. |
| `SubtractVectors(Vector<>,Vector<>)` | Subtracts one vector from another element-wise. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the model-specific training logic for STL decomposition. |
| `TriCube()` | Calculates the tri-cube weight function used in LOESS smoothing. |
| `ValidateDecomposition(Vector<>)` | Validates the decomposition results to ensure they are reasonable. |
| `WeightedLeastSquares(List<ValueTuple<,,>>)` | Calculates a weighted average of points based on their weights. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_residual` | The residual component of the time series. |
| `_seasonal` | The seasonal component of the time series. |
| `_stlOptions` | Configuration options for the STL decomposition. |
| `_trend` | The trend component of the time series. |

