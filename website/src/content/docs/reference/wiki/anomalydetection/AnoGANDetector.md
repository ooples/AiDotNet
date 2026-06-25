---
title: "AnoGANDetector<T>"
description: "Implements AnoGAN (Anomaly Detection with Generative Adversarial Networks)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.NeuralNetwork`

Implements AnoGAN (Anomaly Detection with Generative Adversarial Networks).

## For Beginners

AnoGAN trains a GAN to generate normal data, then detects
anomalies by finding the latent code that best reconstructs a query point.
Points that cannot be well-reconstructed are anomalies.

## How It Works

The algorithm works by:

1. Train a GAN (Generator + Discriminator) on normal data
2. For anomaly scoring, find z that minimizes reconstruction error
3. Anomaly score combines reconstruction loss and discriminator feature loss

**When to use:**

- Image anomaly detection
- When you have only normal examples for training
- High-dimensional data where reconstruction quality matters

**Industry Standard Defaults:**

- Latent dimensions: 64
- Hidden dimensions: 128
- Epochs: 100
- Learning rate: 0.0002
- Contamination: 0.1 (10%)

Reference: Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., and Langs, G. (2017).
"Unsupervised Anomaly Detection with Generative Adversarial Networks." IPMI.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnoGANDetector(Int32,Int32,Int32,Double,Int32,Double,Int32)` | Creates a new AnoGAN anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimensions. |
| `LatentDim` | Gets the latent dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropDiscriminatorFeaturesToInput(Vector<>,Vector<>,Vector<>)` | Backpropagates a gradient on the discriminator's FEATURE layer (h2) to a gradient on the discriminator input, WITHOUT updating weights. |
| `BackpropGeneratorToInput(Vector<>,Vector<>,Vector<>,Vector<>)` | Backpropagates a gradient on the generator OUTPUT to a gradient on the latent input z, WITHOUT updating any weights. |
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

