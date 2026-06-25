---
title: "FrechetInceptionDistance<T>"
description: "Fréchet Inception Distance (FID) - A metric for evaluating the quality of generated images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Fréchet Inception Distance (FID) - A metric for evaluating the quality of generated images.

FID measures how similar generated images are to real images by comparing their
statistical properties in a feature space. Lower FID scores indicate better quality.

The algorithm:

1. Extract features from images using a pre-trained neural network
2. Compute statistics (mean and covariance) for real and generated image features
3. Compute the Fréchet distance between the two Gaussian distributions

Formula: FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
where μ is mean, Σ is covariance, Tr is trace

Typical FID scores:

- FID less than 10: Excellent quality
- FID 10-20: Good quality
- FID 20-50: Moderate quality
- FID greater than 50: Poor quality

Based on "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
by Heusel et al. (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FrechetInceptionDistance(ConvolutionalNeuralNetwork<>,Int32)` | Initializes a new instance of FID calculator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` | Gets the dimensionality of extracted features. |
| `FeatureLayer` | Gets or sets the layer index from which to extract features. |
| `FeatureNetwork` | Gets the feature extraction network used for computing image representations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFID(Tensor<>,Tensor<>)` | Computes the FID score between real and generated images. |
| `ComputeFIDWithStats(Vector<>,Matrix<>,Tensor<>)` | Computes FID using pre-computed statistics for the real distribution. |
| `ComputeFrechetDistance(Vector<>,Matrix<>,Vector<>,Matrix<>)` | Computes the Fréchet distance between two Gaussian distributions. |
| `ComputeStatistics(Matrix<>)` | Computes mean and covariance matrix of feature vectors using vectorized operations. |
| `ComputeTrace(Matrix<>)` | Computes the trace of a matrix (sum of diagonal elements). |
| `ComputeTraceSqrtCovProduct(Matrix<>,Matrix<>)` | Computes Tr(sqrt(cov1 * cov2)) using Newton-Schulz iteration for matrix square root. |
| `ComputeTraceSqrtViaEigenvalues(Matrix<>,Int32)` | Computes trace(sqrt(A)) using eigenvalue decomposition via Jacobi iteration. |
| `ExtractFeatures(Tensor<>)` | Extracts features from images using the feature network. |
| `PrecomputeStatistics(Tensor<>)` | Pre-computes statistics for a set of images. |

