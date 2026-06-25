---
title: "VideoNeuralNetworkBase<T>"
description: "Base class for video-focused neural networks that can operate in both ONNX inference and native training modes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Video`

Base class for video-focused neural networks that can operate in both ONNX inference and native training modes.

## For Beginners

Video neural networks process sequences of image frames to perform
tasks like super-resolution, frame interpolation, optical flow estimation, denoising,
stabilization, and inpainting. This base class provides:

- Support for pre-trained ONNX models (fast inference with existing models)
- Full training capability from scratch (like other neural networks)
- Frame preprocessing utilities (normalization, patch extraction)
- Temporal context handling (multi-frame processing)

You can use derived classes in two ways:

1. Load a pre-trained ONNX model for quick inference
2. Build and train a new model from scratch

## How It Works

This class extends `NeuralNetworkBase` to provide video-specific functionality
while maintaining full integration with the AiDotNet neural network infrastructure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoNeuralNetworkBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the VideoNeuralNetworkBase class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function for this model. |
| `FrameHeight` | Gets or sets the expected frame height for this model. |
| `FrameWidth` | Gets or sets the expected frame width for this model. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `NumChannels` | Gets or sets the number of color channels expected by this model. |
| `NumFrames` | Gets or sets the number of frames this model processes at once. |
| `OnnxDecoder` | Gets or sets the ONNX decoder model (for encoder-decoder architectures). |
| `OnnxEncoder` | Gets or sets the ONNX encoder model (for encoder-decoder architectures). |
| `OnnxModel` | Gets or sets the ONNX model (for single-model architectures). |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BilinearSample(Tensor<>,Int32,Int32,Double,Double,Boolean,Int32,Int32,Int32)` | Performs bilinear sampling from a feature tensor. |
| `ConcatenateFeatures(Tensor<>,Tensor<>)` | Concatenates two feature tensors along the channel dimension. |
| `DenormalizeFrames(Tensor<>)` | Denormalizes frame values from [0, 1] back to [0, 255]. |
| `Dispose(Boolean)` | Disposes of resources used by this model. |
| `ExtractFrame(Tensor<>,Int32)` | Extracts a single frame from a multi-frame tensor. |
| `Forward(Tensor<>)` | Performs a forward pass through the native neural network layers. |
| `GetFeatureValue(Tensor<>,Int32,Int32,Int32,Int32,Boolean,Int32,Int32,Int32)` | Gets a value from a feature tensor at the specified position. |
| `NormalizeFrames(Tensor<>)` | Normalizes frame pixel values from [0, 255] to [0, 1]. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PreprocessFrames(Tensor<>)` | Preprocesses raw video frames for model input. |
| `RunOnnxInference(Tensor<>)` | Runs inference using ONNX model(s). |
| `StoreFrame(Tensor<>,Tensor<>,Int32)` | Stores a single frame into a multi-frame tensor. |
| `WarpFeature(Tensor<>,Tensor<>)` | Warps a feature map using optical flow via bilinear interpolation. |

