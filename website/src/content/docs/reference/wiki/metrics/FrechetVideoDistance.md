---
title: "FrechetVideoDistance<T>"
description: "Fréchet Video Distance (FVD) - A metric for evaluating the quality of generated videos."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Fréchet Video Distance (FVD) - A metric for evaluating the quality of generated videos.

## How It Works

FVD extends Fréchet Inception Distance (FID) to videos by using a 3D video feature extractor
(typically Inflated 3D ConvNet, I3D) to compare the distribution of generated videos
against real videos.

The algorithm:

1. Extract spatiotemporal features from video clips using a 3D CNN
2. Compute statistics (mean and covariance) for real and generated video features
3. Compute the Fréchet distance between the two Gaussian distributions

Formula: FVD = ||mu_1 - mu_2||^2 + Tr(Sigma_1 + Sigma_2 - 2 * sqrt(Sigma_1 * Sigma_2))

Typical FVD scores:

- FVD < 50: Excellent quality (hard to distinguish from real)
- FVD 50-100: Good quality
- FVD 100-300: Moderate quality
- FVD > 300: Poor quality

Based on "Towards Accurate Generative Models of Video: A New Metric and Challenges"
by Unterthiner et al. (2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FrechetVideoDistance(ConvolutionalNeuralNetwork<>,Int32,Int32)` | Initializes a new instance of FVD calculator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` | Gets the dimensionality of extracted video features. |
| `FeatureNetwork` | Gets the 3D feature extraction network (e.g., I3D) used for video representations. |
| `FramesPerClip` | Gets or sets the number of frames per video clip for feature extraction. |
| `SamplingStrategy` | Gets or sets the frame sampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFVD(Tensor<>,Tensor<>)` | Computes the FVD score between real and generated videos. |
| `ComputeFVDWithStats(Vector<>,Matrix<>,Tensor<>)` | Computes FVD using pre-computed statistics for the real distribution. |
| `ComputeFrechetDistance(Vector<>,Matrix<>,Vector<>,Matrix<>)` | Computes the Fréchet distance between two Gaussian distributions. |
| `ComputeStatistics(Matrix<>)` | Computes mean and covariance matrix of feature vectors. |
| `ComputeTrace(Matrix<>)` | Computes the trace of a matrix. |
| `ComputeTraceSqrtCovProduct(Matrix<>,Matrix<>)` | Computes Tr(sqrt(cov1 * cov2)) using Newton-Schulz iteration. |
| `ExtractVideoClip(Tensor<>,Int32,Boolean)` | Extracts a single video clip from the batch tensor. |
| `ExtractVideoFeatures(Tensor<>)` | Extracts features from videos using the 3D feature network. |
| `GlobalAveragePool(Tensor<>)` | Applies global average pooling to reduce spatial and temporal dimensions. |
| `PrecomputeStatistics(Tensor<>)` | Pre-computes statistics for a set of videos. |
| `SampleFrameIndices(Int32)` | Samples frame indices according to the sampling strategy. |

