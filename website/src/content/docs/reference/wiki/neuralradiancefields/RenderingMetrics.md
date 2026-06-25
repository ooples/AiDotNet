---
title: "RenderingMetrics<T>"
description: "Provides image quality metrics for evaluating neural rendering methods like NeRF."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.NeuralRadianceFields.Metrics`

Provides image quality metrics for evaluating neural rendering methods like NeRF.

## For Beginners

When a neural network generates an image (like a new view
of a 3D scene), we need ways to measure how good that image is compared to a real
photograph. These metrics give us numbers that tell us how similar two images are.

## How It Works

These metrics are essential for evaluating the quality of rendered images from
neural radiance fields, 3D Gaussian splatting, and other view synthesis methods.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEdgeMagnitude(Tensor<>)` | Computes edge magnitude using Sobel-like operators. |
| `ComputeGlobalSSIM(Tensor<>,Tensor<>,Double,Double)` | Computes global SSIM when image is too small for windowed approach. |
| `ComputeLocalStatistics(Tensor<>)` | Computes local statistics (mean, variance) as feature maps. |
| `ComputeMSE(Tensor<>,Tensor<>)` | Computes Mean Squared Error. |
| `ComputeWindowSSIM(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32,Double,Double)` | Computes SSIM for a single window. |
| `GetPixel(Tensor<>,Int32,Int32,Int32,Int32)` | Gets a pixel value from an image tensor. |
| `MAE(Tensor<>,Tensor<>)` | Computes Mean Absolute Error between two tensors. |
| `MSE(Tensor<>,Tensor<>)` | Computes Mean Squared Error between two tensors. |
| `PSNR(Tensor<>,Tensor<>,Double)` | Computes Peak Signal-to-Noise Ratio (PSNR) between two images. |
| `SSIM(Tensor<>,Tensor<>,Int32,Double,Double,Double)` | Computes Structural Similarity Index Measure (SSIM) between two images. |
| `SimplifiedLPIPS(Tensor<>,Tensor<>)` | Computes a simplified perceptual loss (L1 in feature space) as a proxy for LPIPS. |
| `ValidateShapes(Tensor<>,Tensor<>)` | Validates that two tensors have the same shape. |

