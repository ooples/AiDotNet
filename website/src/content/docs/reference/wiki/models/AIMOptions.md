---
title: "AIMOptions<T>"
description: "Configuration options for AIM (Adaptive Iterative Mechanism), a marginal-based differentially private synthetic data generation method."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for AIM (Adaptive Iterative Mechanism), a marginal-based
differentially private synthetic data generation method.

## For Beginners

AIM is a non-neural approach to private data synthesis:

1. Pick important "statistics" about the data (marginals = histograms of 1-3 columns)
2. Measure them with added noise (for privacy)
3. Create fake data that matches those noisy statistics

Unlike GAN/VAE models, AIM uses mathematical optimization (not deep learning),
which often works better for smaller datasets and provides formal privacy guarantees.

Example:

## How It Works

AIM generates synthetic data through iterative marginal measurements:

- **Marginal selection**: Uses the exponential mechanism to privately select informative marginals
- **Noisy measurement**: Measures selected marginals with calibrated Gaussian noise
- **Synthetic optimization**: Iteratively refines synthetic data to match measured marginals

Reference: "AIM: An Adaptive and Iterative Mechanism for Differentially Private
Synthetic Data" (McKenna et al., 2022)

## Properties

| Property | Summary |
|:-----|:--------|
| `Epsilon` | Gets or sets the total privacy budget (epsilon). |
| `LearningRate` | Gets or sets the learning rate for synthetic data optimization. |
| `MarginalsPerIteration` | Gets or sets the number of marginals to select per iteration. |
| `MaxMarginalOrder` | Gets or sets the maximum marginal order to consider. |
| `NumBins` | Gets or sets the number of bins for discretizing continuous columns. |
| `NumIterations` | Gets or sets the number of iterations for synthetic data optimization. |

