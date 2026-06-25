---
title: "KernelInceptionDistance<T>"
description: "Kernel Inception Distance (KID) - A metric for evaluating the quality of generated images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Kernel Inception Distance (KID) - A metric for evaluating the quality of generated images.

## How It Works

KID measures how similar generated images are to real images using the Maximum Mean Discrepancy (MMD)
with a polynomial kernel in the feature space of an Inception network.

Advantages over FID:

- Unbiased estimator (FID is biased for small sample sizes)
- Provides variance estimates for the metric
- Works well with smaller datasets
- More robust to sample size variations

Formula: KID = MMD^2(F_real, F_generated) using polynomial kernel k(x,y) = (x^T y / d + 1)^3

Typical KID scores (multiplied by 1000 for readability):

- KID < 0.5: Excellent quality
- KID 0.5-2.0: Good quality
- KID 2.0-5.0: Moderate quality
- KID > 5.0: Poor quality

Based on "Demystifying MMD GANs" by Binkowski et al. (2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KernelInceptionDistance(ConvolutionalNeuralNetwork<>,Int32,Int32,Int32,Int32)` | Initializes a new instance of KID calculator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` | Gets the dimensionality of extracted features. |
| `FeatureNetwork` | Gets the feature extraction network used for computing image representations. |
| `NumSubsets` | Gets or sets the number of subsets for computing variance estimates. |
| `PolynomialDegree` | Gets or sets the polynomial degree for the kernel. |
| `SubsetSize` | Gets or sets the subset size for variance computation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDotProduct(Matrix<>,Int32,Matrix<>,Int32)` | Computes dot product between two feature vectors. |
| `ComputeKID(Tensor<>,Tensor<>)` | Computes the KID score between real and generated images. |
| `ComputeKIDWithFeatures(Matrix<>,Tensor<>)` | Computes KID using pre-computed feature statistics. |
| `ComputeKIDWithVariance(Tensor<>,Tensor<>)` | Computes KID score with variance estimate using subset sampling. |
| `ComputeMMD(Matrix<>,Matrix<>)` | Computes Maximum Mean Discrepancy (MMD) using polynomial kernel. |
| `ExtractFeatures(Tensor<>)` | Extracts features from images using the feature network. |
| `PrecomputeFeatures(Tensor<>)` | Pre-computes features for a set of images for efficient repeated evaluation. |
| `SampleSubset(Matrix<>,Int32,Random)` | Samples a random subset of features. |

