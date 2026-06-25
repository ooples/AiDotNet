---
title: "DocumentNeuralNetworkBase<T>"
description: "Base class for document-focused neural networks that can operate in both ONNX inference and native training modes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Document`

Base class for document-focused neural networks that can operate in both ONNX inference and native training modes.

## For Beginners

Document neural networks process images of documents (scanned pages, PDFs, photos).
This base class provides:

- Support for pre-trained ONNX models (fast inference with existing models)
- Full training capability from scratch (like other neural networks)
- Document preprocessing utilities (normalization, resizing, etc.)
- Layout-aware feature extraction
- Integration with text encoding for layout-aware models

You can use this class in two ways:

1. Load a pre-trained ONNX model for quick inference
2. Build and train a new model from scratch

## How It Works

This class extends `NeuralNetworkBase` to provide document-specific functionality
while maintaining full integration with the AiDotNet neural network infrastructure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocumentNeuralNetworkBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the DocumentNeuralNetworkBase class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function for this model. |
| `ImageSize` | Gets the expected input image size for this model. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MaxSequenceLength` | Gets the maximum text sequence length for layout-aware models. |
| `OnnxDecoder` | Gets or sets the ONNX decoder model (for encoder-decoder architectures). |
| `OnnxEncoder` | Gets or sets the ONNX encoder model (for encoder-decoder architectures). |
| `OnnxModel` | Gets or sets the ONNX model (for single-model architectures). |
| `PreprocessingTransformer` | Gets or sets the instance-level preprocessing transformer for this document model. |
| `RequiresOCR` | Gets whether this model requires OCR preprocessing. |
| `SupportedDocumentTypes` | Gets the supported document types for this model. |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies industry-standard postprocessing defaults for this specific model type. |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies industry-standard preprocessing defaults for this specific model type. |
| `Dispose(Boolean)` | Disposes of resources used by this model. |
| `EnsureBatchDimension(Tensor<>)` | Adds a batch dimension to a 3D tensor if needed. |
| `Forward(Tensor<>)` | Performs a forward pass through the native neural network layers. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PreprocessDocument(Tensor<>)` | Preprocesses a raw document image for model input. |
| `RunOnnxInference(Tensor<>)` | Runs inference using ONNX model(s). |
| `SafeSerialize` | Validates that an input image tensor has the correct shape. |

