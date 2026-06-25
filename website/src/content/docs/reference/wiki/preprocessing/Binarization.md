---
title: "Binarization<T>"
description: "Binarization - Document binarization with multiple algorithms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Document`

Binarization - Document binarization with multiple algorithms.

## For Beginners

Binarization separates text from background by converting
each pixel to either black or white:

- Otsu: Global threshold, works well for uniform lighting
- Sauvola: Local adaptive, handles varying illumination
- Niblack: Local adaptive, good for degraded documents
- Fixed: Simple fixed threshold value

Key features:

- Multiple binarization algorithms
- Handles varying lighting conditions
- Works with degraded documents
- Configurable parameters

Example usage:

## How It Works

Binarization converts grayscale document images to binary (black and white),
which is essential for many OCR and document analysis tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Binarization` | Creates a new Binarization instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the binarization utility. |
| `FixedThreshold(Tensor<>,Double)` | Applies a fixed threshold. |
| `NiblackBinarization(Tensor<>,Int32,Double)` | Applies Niblack's local adaptive thresholding. |
| `OtsuBinarization(Tensor<>)` | Applies Otsu's method for global thresholding. |
| `Process(Tensor<>,BinarizationMethod)` | Applies binarization to a document image. |
| `SauvolaBinarization(Tensor<>,Int32,Double,Double)` | Applies Sauvola's local adaptive thresholding. |

