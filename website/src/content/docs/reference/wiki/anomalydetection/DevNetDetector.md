---
title: "DevNetDetector<T>"
description: "Implements DevNet (Deep Anomaly Detection Network) for end-to-end anomaly scoring."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.NeuralNetwork`

Implements DevNet (Deep Anomaly Detection Network) for end-to-end anomaly scoring.

## For Beginners

DevNet learns to directly output anomaly scores using a deviation
network. It combines feature learning and anomaly scoring in a single network,
using reference points and deviation loss for training.

## How It Works

The algorithm works by:

1. Learn feature representations through neural network
2. Use Gaussian reference points to define normalcy
3. Train with deviation loss to produce anomaly scores directly

**When to use:**

- When you want end-to-end anomaly scoring
- Tabular data with known anomaly labels (semi-supervised)
- When reconstruction-based methods don't work well

**Industry Standard Defaults:**

- Hidden dimensions: 64
- Output dimensions: 1
- Epochs: 50
- Learning rate: 0.001
- Contamination: 0.1 (10%)

Reference: Pang, G., Shen, C., and van den Hengel, A. (2019).
"Deep Anomaly Detection with Deviation Networks." KDD.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DevNetDetector(Int32,Int32,Double,Double,Double,Int32)` | Creates a new DevNet anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

