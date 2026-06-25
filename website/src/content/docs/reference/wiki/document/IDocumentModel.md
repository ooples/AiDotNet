---
title: "IDocumentModel<T>"
description: "Base interface for all document AI models in AiDotNet."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Base interface for all document AI models in AiDotNet.

## For Beginners

A document AI model processes document images (scanned pages, PDFs, photos of text)
to extract information, understand layout, or answer questions.

Key concepts:

- Document images have shape [batch, channels, height, width]
- Models can run in Native mode (pure C#) or ONNX mode (optimized runtime)
- All models support both training and inference
- Many document models combine vision and language understanding

Example usage:

## How It Works

This interface extends `IFullModel` to provide the core contract
for document AI models, inheriting standard methods for training, inference, model persistence,
and gradient computation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` | Gets the expected input image size (assumes square images). |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MaxSequenceLength` | Gets the maximum sequence length for text processing. |
| `RequiresOCR` | Gets whether this model requires OCR preprocessing. |
| `SupportedDocumentTypes` | Gets the supported document types for this model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EncodeDocument(Tensor<>)` | Processes a document image and returns encoded features. |
| `GetModelSummary` | Gets a summary of the model architecture. |
| `ValidateInputShape(Tensor<>)` | Validates that an input tensor has the correct shape for this model. |

