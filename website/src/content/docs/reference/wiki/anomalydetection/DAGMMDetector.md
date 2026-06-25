---
title: "DAGMMDetector<T>"
description: "Implements DAGMM (Deep Autoencoding Gaussian Mixture Model) for anomaly detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.NeuralNetwork`

Implements DAGMM (Deep Autoencoding Gaussian Mixture Model) for anomaly detection.

## For Beginners

DAGMM combines an autoencoder with a Gaussian Mixture Model
in an end-to-end trainable architecture. The autoencoder learns compressed representations,
and the GMM learns the density of normal data in the latent space.

## How It Works

The algorithm works by:

1. Compression network (autoencoder) learns latent representation
2. Estimation network predicts GMM membership probabilities
3. GMM models the distribution of latent codes + reconstruction features
4. Anomaly score is the negative log-likelihood under the GMM

**When to use:**

- Complex multivariate data
- When you want to model multiple modes of normal behavior
- When reconstruction error alone is insufficient

**Industry Standard Defaults:**

- Latent dimensions: 4
- Hidden dimensions: 64
- Number of mixtures: 4
- Epochs: 100
- Contamination: 0.1 (10%)

Reference: Zong, B., et al. (2018).
"Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection." ICLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DAGMMDetector(Int32,Int32,Int32,Int32,Double,Double,Int32)` | Creates a new DAGMM anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LatentDim` | Gets the latent dimensions. |
| `NumMixtures` | Gets the number of GMM components. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

