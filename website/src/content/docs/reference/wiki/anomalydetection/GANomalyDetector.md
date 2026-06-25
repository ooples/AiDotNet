---
title: "GANomalyDetector<T>"
description: "Implements GANomaly for anomaly detection using GAN-based reconstruction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.NeuralNetwork`

Implements GANomaly for anomaly detection using GAN-based reconstruction.

## For Beginners

GANomaly learns to encode, decode, and re-encode data.
Anomalies are detected when the encoding of the original differs from the
encoding of the reconstruction, indicating the model cannot properly represent the data.

## How It Works

The algorithm works by:

1. Encoder maps input to latent space z
2. Decoder reconstructs from z
3. Second encoder re-encodes the reconstruction
4. Anomaly score is the difference between original encoding and re-encoding

**When to use:**

- Image anomaly detection
- When reconstruction error alone is insufficient
- Semi-supervised anomaly detection with only normal examples

**Industry Standard Defaults:**

- Latent dimensions: 32
- Hidden dimensions: 64
- Epochs: 100
- Learning rate: 0.0002
- Contamination: 0.1 (10%)

Reference: Akcay, S., Atapour-Abarghouei, A., and Breckon, T. P. (2018).
"GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training." ACCV.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GANomalyDetector(Int32,Int32,Int32,Double,Double,Int32)` | Creates a new GANomaly anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimensions. |
| `LatentDim` | Gets the latent dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

