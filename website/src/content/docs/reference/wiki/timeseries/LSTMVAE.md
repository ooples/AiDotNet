---
title: "LSTMVAE<T>"
description: "Implements LSTM-VAE (Long Short-Term Memory Variational Autoencoder) for anomaly detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries.AnomalyDetection`

Implements LSTM-VAE (Long Short-Term Memory Variational Autoencoder) for anomaly detection.

## For Beginners

LSTM-VAE is like a compression and decompression system for time series:

1. The encoder "compresses" your time series into a simpler representation
2. The decoder tries to "decompress" it back to the original
3. For normal patterns, this works well (low reconstruction error)
4. For anomalies, the reconstruction is poor (high error) because the model hasn't seen such patterns

Think of it like a photocopier that's been trained on normal documents - it copies normal
pages perfectly but produces poor copies of unusual documents, making them easy to identify.

## How It Works

LSTM-VAE combines the sequential modeling capabilities of LSTMs with the probabilistic
framework of Variational Autoencoders. It learns a compressed latent representation
of normal time series patterns and detects anomalies as points with high reconstruction error.

Key components:

- LSTM Encoder: Compresses time series into latent space
- Latent Space: Probabilistic representation (mean and variance)
- LSTM Decoder: Reconstructs time series from latent representation
- Anomaly Detection: Based on reconstruction error and KL divergence

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.TimeSeries.AnomalyDetection;
using AiDotNet.Tensors.LinearAlgebra;

double[] series =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};
var x = new Matrix<double>(series.Length, 1);
for (int i = 0; i < series.Length; i++) x[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LSTMVAE<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"LSTMVAE: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTMVAE(LSTMVAEOptions<>)` | Initializes a new instance of the LSTMVAE class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAnomalyScores(Matrix<>)` | Computes anomaly scores for a time series. |
| `DetectAnomalies(Matrix<>)` | Detects anomalies in a time series using reconstruction error. |
| `GetOptions` |  |

