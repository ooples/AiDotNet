---
title: "SpectralAnalysisModel<T>"
description: "Implements spectral analysis for time series data, which transforms time domain signals into the frequency domain."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements spectral analysis for time series data, which transforms time domain signals into the frequency domain.

## For Beginners

Spectral analysis is like breaking down a song into its individual notes. Just as a song is made up of different
notes played at different times, time series data can contain different patterns that repeat at different frequencies.

For example, if you analyze temperature data over several years, spectral analysis might reveal:

- A strong yearly cycle (frequency = 1/365 days) due to seasonal changes
- A daily cycle (frequency = 1/24 hours) due to day/night temperature differences
- Other cycles you might not notice just by looking at the raw data

This model uses the Fast Fourier Transform (FFT) algorithm to convert time data into frequency information,
showing you how strong each frequency component is in your data.

## How It Works

Spectral analysis is a technique used to analyze the frequency content of time series data. It helps identify
periodic patterns and dominant frequencies in the data by transforming it from the time domain to the frequency domain.

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
    .ConfigureModel(new SpectralAnalysisModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"SpectralAnalysisModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralAnalysisModel(SpectralAnalysisOptions<>)` | Initializes a new instance of the SpectralAnalysisModel class with optional configuration options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWindowFunction(Vector<>)` | Applies a window function to the input signal to reduce spectral leakage. |
| `ComputeFFT(Vector<>,Int32)` | Computes the Fast Fourier Transform (FFT) of the input signal. |
| `CreateInstance` | Creates a new instance of the SpectralAnalysisModel with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the performance of the trained model on test data. |
| `FFT(Vector<Complex<>>)` | Recursively computes the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm. |
| `GetFrequencies` | Gets the frequency values corresponding to each point in the periodogram. |
| `GetModelMetadata` | Gets metadata about the model, including its type, parameters, and characteristics of the spectral analysis. |
| `GetPeriodogram` | Gets the periodogram (power spectral density) values for each frequency. |
| `Predict(Matrix<>)` | Returns the periodogram (power spectral density) as the prediction result. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the dominant frequency component. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the spectral analysis model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_frequencies` | The frequency values corresponding to each point in the periodogram. |
| `_periodogram` | The power spectral density (periodogram) values for each frequency. |
| `_spectralOptions` | Configuration options for the spectral analysis model. |

