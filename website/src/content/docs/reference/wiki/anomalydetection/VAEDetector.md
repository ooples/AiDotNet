---
title: "VAEDetector<T>"
description: "Detects anomalies using Variational Autoencoder (VAE)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.NeuralNetwork`

Detects anomalies using Variational Autoencoder (VAE).

## For Beginners

A VAE is a generative neural network that learns to encode data into
a lower-dimensional probabilistic latent space and decode it back. Anomalies are points
that are poorly reconstructed or fall in low-probability regions of the latent space.

## How It Works

The algorithm works by:

1. Train encoder to map data to latent distribution (mean + variance)
2. Train decoder to reconstruct from latent samples
3. Score combines reconstruction error and KL divergence

**When to use:**

- Complex, high-dimensional data
- When you want probabilistic anomaly scores
- Image, text, or structured data anomalies

**Industry Standard Defaults:**

- Latent dimensions: 10
- Hidden dimensions: 64
- Learning rate: 0.001
- Epochs: 100
- Contamination: 0.1 (10%)

Reference: Kingma, D.P., Welling, M. (2014). "Auto-Encoding Variational Bayes." ICLR.
An, J., Cho, S. (2015). "Variational Autoencoder based Anomaly Detection."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VAEDetector(Int32,Int32,Int32,Double,Double,Int32)` | Creates a new VAE anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden layer dimensions. |
| `LatentDim` | Gets the latent space dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

