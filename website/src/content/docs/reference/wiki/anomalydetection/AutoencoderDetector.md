---
title: "AutoencoderDetector<T>"
description: "Implements an Autoencoder-based method for anomaly detection using reconstruction error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.NeuralNetwork`

Implements an Autoencoder-based method for anomaly detection using reconstruction error.

## For Beginners

An autoencoder is a neural network that learns to compress data into
a smaller representation and then reconstruct it. Normal data can be reconstructed well,
but anomalies (which the autoencoder hasn't learned to represent) will have high reconstruction error.

## How It Works

The algorithm works by:

1. Training a simple autoencoder (encoder-decoder network) on the data
2. For each data point, computing the reconstruction error (how different the output is from input)
3. Points with high reconstruction error are likely anomalies

**When to use:** Autoencoder-based detection is particularly effective for:

- High-dimensional data
- Data with complex patterns
- When you want the detector to automatically learn important features

**Industry Standard Defaults:**

- Encoding dimension: Auto (input_dim/2)
- Epochs: 50
- Learning rate: 0.01
- Batch size: 32
- Contamination: 0.1 (10%)

This implementation uses a simple fully-connected autoencoder with one hidden layer.
For more complex scenarios, consider using a deeper architecture.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoencoderDetector(Int32,Int32,Double,Int32,Double,Int32)` | Creates a new Autoencoder-based anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EncodingDim` | Gets the encoding dimension (bottleneck size). |
| `Epochs` | Gets the number of training epochs. |
| `LearningRate` | Gets the learning rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Matrix<>)` | Gets the encoded representation of the input data. |
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

