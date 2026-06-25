---
title: "NoiseRemoval<T>"
description: "NoiseRemoval - Document image noise removal with multiple algorithms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Document`

NoiseRemoval - Document image noise removal with multiple algorithms.

## For Beginners

Scanned documents often have noise (random spots, grain).
This tool removes noise while keeping text clear:

- Median: Best for salt-and-pepper noise
- Gaussian: General smoothing
- Bilateral: Edge-preserving smoothing
- Morphological: Removes small artifacts

Key features:

- Multiple noise removal algorithms
- Edge-preserving options
- Configurable filter sizes
- Works with binary and grayscale images

Example usage:

## How It Works

NoiseRemoval provides various filtering techniques to clean up document images
by reducing noise while preserving important features like text edges.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NoiseRemoval` | Creates a new NoiseRemoval instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BilateralFilter(Tensor<>,Int32,Double,Double)` | Applies bilateral filtering for edge-preserving smoothing. |
| `Dilate(Tensor<>,Int32)` | Applies morphological dilation. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the noise removal utility. |
| `Erode(Tensor<>,Int32)` | Applies morphological erosion. |
| `GaussianBlur(Tensor<>,Int32,Double)` | Applies Gaussian blur for general noise reduction. |
| `MedianFilter(Tensor<>,Int32)` | Applies median filtering for salt-and-pepper noise removal. |
| `MorphologicalClosing(Tensor<>,Int32)` | Applies morphological closing (dilation followed by erosion) for hole filling. |
| `MorphologicalOpening(Tensor<>,Int32)` | Applies morphological opening (erosion followed by dilation) for artifact removal. |
| `Process(Tensor<>,NoiseRemovalMethod)` | Applies noise removal to a document image. |

