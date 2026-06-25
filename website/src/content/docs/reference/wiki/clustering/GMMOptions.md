---
title: "GMMOptions<T>"
description: "Configuration options for Gaussian Mixture Model clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Gaussian Mixture Model clustering.

## For Beginners

GMM is like finding overlapping groups in data.

Imagine drops of different colored paint on paper:

- K-Means: Draws hard boundaries between colors
- GMM: Allows colors to blend, giving probability of belonging to each

Key features:

- Soft clustering: Each point has probabilities for all clusters
- Captures different cluster shapes and sizes
- Based on statistical modeling

When to use GMM:

- Clusters have different sizes or shapes
- You need probability of cluster membership
- Data might have overlapping clusters

## How It Works

GMM assumes data is generated from a mixture of several Gaussian distributions.
It finds the parameters of these distributions using the Expectation-Maximization (EM) algorithm.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GMMOptions` | Initializes a new instance of GMMOptions with GMM-appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowLowWeights` | Gets or sets whether to allow components with very low weights. |
| `ComputeLowerBound` | Gets or sets whether to compute the lower bound during training. |
| `CovarianceType` | Gets or sets the type of covariance parameters to use. |
| `InitMethod` | Gets or sets the initialization method. |
| `MinWeight` | Gets or sets the minimum weight threshold for components. |
| `NumComponents` | Gets or sets the number of mixture components. |
| `RegularizationCovariance` | Gets or sets the regularization added to the diagonal of covariance. |
| `WeightConcentrationPrior` | Gets or sets the weight concentration prior (for Dirichlet process). |

