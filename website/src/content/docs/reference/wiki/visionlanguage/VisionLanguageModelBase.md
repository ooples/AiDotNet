---
title: "VisionLanguageModelBase<T>"
description: "Base class for vision-language neural networks that can operate in both ONNX inference and native training modes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.VisionLanguage`

Base class for vision-language neural networks that can operate in both ONNX inference and native training modes.

## For Beginners

Vision-language models process both images and text together. This base class provides:

- Support for pre-trained ONNX models (fast inference with existing models)
- Full training capability from scratch (like other neural networks)
- Image preprocessing utilities (normalization, resizing)
- Dual-encoder architecture support (separate image and text encoders)

You can use this class in two ways:

1. Load a pre-trained ONNX model for quick inference
2. Build and train a new model from scratch

## How It Works

This class extends `NeuralNetworkBase` to provide vision-language-specific functionality
while maintaining full integration with the AiDotNet neural network infrastructure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisionLanguageModelBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the VisionLanguageModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function for this model. |
| `EmbeddingDim` | Gets the embedding dimensionality for this model. |
| `ImageChannels` | Gets the number of image channels expected (typically 3 for RGB). |
| `ImageSize` | Gets the expected input image size (height = width in pixels). |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `OnnxImageEncoder` | Gets or sets the ONNX image encoder model (for dual-encoder architectures). |
| `OnnxModel` | Gets or sets the ONNX model (for single-model architectures). |
| `OnnxTextEncoder` | Gets or sets the ONNX text encoder model (for dual-encoder architectures). |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeVisionLanguageBoundary(Int32,Int32,Int32,Int32,Int32,Int32)` | Computes an encoder/decoder split boundary from repeated block counts. |
| `ComputeVisualPatchSize(Int32,Int32,Boolean)` | Computes a patch size from an image size and target visual-token budget. |
| `CosineSimilarity(Tensor<>,Tensor<>)` | Computes cosine similarity between two embedding tensors. |
| `Dispose(Boolean)` | Disposes of resources used by this model. |
| `EnumerateAllAuxiliaryTrainableLayers` | Combined helper that yields trainable-layer references for both the dual-stream `TextEncoderLayers` and any registered auxiliary streams. |
| `EnumerateAuxiliaryStreamTrainableLayers` | Iterates the registered auxiliary streams (Q-Former, decoder, fusion bridge, etc.) yielding each layer's `LayerBase` view. |
| `EnumerateTextEncoderTrainableLayers` | Helper for subclasses overriding `GetExtraTrainableLayers` to surface their `TextEncoderLayers` to the base weight-registry walker. |
| `L2Normalize(Tensor<>)` | L2-normalizes an embedding tensor. |
| `NormalizeImage(Tensor<>,Double[],Double[])` | Normalizes an image tensor using ImageNet mean and standard deviation. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PreprocessImage(Tensor<>)` | Preprocesses a raw image tensor for model input. |
| `RegisterAuxiliaryEncoderStream(List<ILayer<>>)` | Registers an auxiliary encoder stream that lives outside `Layers`. |
| `ResamplerBlockLayerCount(Double)` | Returns the layer count contributed by a cross-attention resampler block. |
| `Softmax(Tensor<>)` | Applies softmax to convert logits to probabilities. |
| `SplitDualStreamLayers(IEnumerable<ILayer<>>,Int32)` | Splits an OpenCLIP-shaped layer factory output (vision pre-norm + N×vision-block + vision-projection + text pre-norm + N×text-block + text-projection) into the model's `Layers` list (vision portion) and `TextEncoderLayers` (text portion). |
| `TransformerBlockLayerCount(Double)` | Returns the layer count contributed by a standard transformer block. |
| `ValidateEncoderDecoderBoundary(Int32)` | Verifies the computed encoder/decoder split is inside the current layer list. |
| `ValidateVisualPatchOptions(Int32,Int32)` | Validates image/token settings used to derive patch sizes for native VLM layer factories. |

## Fields

| Field | Summary |
|:-----|:--------|
| `TextEncoderLayers` | Text-encoder layers, walked by `EncodeText` in subclasses. |

