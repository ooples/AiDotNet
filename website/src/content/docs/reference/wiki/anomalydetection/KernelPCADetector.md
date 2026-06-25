---
title: "KernelPCADetector<T>"
description: "Detects anomalies using Kernel PCA reconstruction error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Linear`

Detects anomalies using Kernel PCA reconstruction error.

## For Beginners

Kernel PCA extends PCA to capture non-linear patterns by mapping data
to a higher-dimensional feature space using a kernel function. Anomalies have high
reconstruction error when projected back from this space.

## How It Works

The algorithm works by:

1. Compute kernel matrix (e.g., RBF kernel)
2. Center the kernel matrix
3. Extract principal components in kernel space
4. Compute reconstruction error for each point

**When to use:**

- Non-linear relationships in data
- When linear PCA doesn't capture patterns well
- Complex, curved cluster structures

**Industry Standard Defaults:**

- Kernel: RBF (Gaussian)
- Gamma: 1/n_features (auto)
- Number of components: 0.95 (95% variance)
- Contamination: 0.1 (10%)

Reference: Schölkopf, B., Smola, A., Müller, K.R. (1998). "Nonlinear Component Analysis
as a Kernel Eigenvalue Problem." Neural Computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KernelPCADetector(Double,Double,KernelPCADetector<>.KernelType,Double,Int32)` | Creates a new Kernel PCA anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gamma` | Gets the gamma parameter for RBF kernel. |
| `Kernel` | Gets the kernel type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

