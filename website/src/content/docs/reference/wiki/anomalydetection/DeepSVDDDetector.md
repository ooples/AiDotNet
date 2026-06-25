---
title: "DeepSVDDDetector<T>"
description: "Detects anomalies using Deep SVDD (Support Vector Data Description)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.NeuralNetwork`

Detects anomalies using Deep SVDD (Support Vector Data Description).

## For Beginners

Deep SVDD trains a neural network to map normal data points close to
a hypersphere center in the output space. Anomalies are points that map far from this center.
It combines deep learning with the classic SVDD concept.

## How It Works

The algorithm works by:

1. Initialize network and compute center from initial encodings
2. Train network to minimize distance of normal points to center
3. Anomaly score is the distance to the center

**When to use:**

- One-class classification with deep learning
- When you have only normal examples for training
- Complex, high-dimensional data

**Industry Standard Defaults:**

- Hidden dimensions: 64
- Output dimensions: 32
- Learning rate: 0.001
- Epochs: 100
- Contamination: 0.1 (10%)

Reference: Ruff, L., et al. (2018). "Deep One-Class Classification." ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepSVDDDetector(Int32,Int32,Int32,Double,Double,Int32)` | Creates a new Deep SVDD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden layer dimensions. |
| `OutputDim` | Gets the output dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

