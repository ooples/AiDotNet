---
title: "DocumentPreprocessor<T>"
description: "DocumentPreprocessor - Comprehensive document image preprocessing pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Document`

DocumentPreprocessor - Comprehensive document image preprocessing pipeline.

## For Beginners

Document preprocessing improves model accuracy by:

- Normalizing image characteristics
- Removing noise and artifacts
- Correcting geometric distortions
- Enhancing text contrast

Key features:

- Chained preprocessing pipeline
- Configurable operation order
- Quality-aware preprocessing
- Batch processing support

Example usage:

## How It Works

DocumentPreprocessor provides a unified interface for applying multiple
preprocessing operations to document images before feeding them to AI models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocumentPreprocessor` | Creates a DocumentPreprocessor with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CenterCrop(Tensor<>,Int32,Int32)` | Center crops an image to the specified dimensions. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the preprocessor. |
| `Pad(Tensor<>,Int32,Int32,)` | Pads an image to the specified dimensions. |
| `Preprocess(Tensor<>,DocumentPreprocessingOptions)` | Applies the full preprocessing pipeline to a document image. |
| `PreprocessBatch(IEnumerable<Tensor<>>,DocumentPreprocessingOptions)` | Applies preprocessing to multiple document images. |
| `Resize(Tensor<>,Int32,Int32,InterpolationMethod)` | Resizes an image to the specified dimensions. |
| `ToGrayscale(Tensor<>)` | Converts an RGB image to grayscale. |

