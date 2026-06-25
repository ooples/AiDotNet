---
title: "SegmentationModelBase<T>"
description: "Abstract base class for all segmentation models, providing common dual-mode (native + ONNX) infrastructure, batch handling, forward/backward passes, and serialization."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for all segmentation models, providing common dual-mode (native + ONNX)
infrastructure, batch handling, forward/backward passes, and serialization.

## For Beginners

This is the foundation for all segmentation models in the library.
It handles the plumbing that every segmentation model needs:

- Loading pre-trained ONNX models for fast inference
- Native mode for training from scratch or fine-tuning
- Converting images between batched and unbatched formats
- Saving and loading model weights

You don't use this class directly — instead, create a concrete model like SegFormer, Mask2Former,
or SAM that extends this base class.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegmentationModelBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32)` | Initializes the base in native (trainable) mode. |
| `SegmentationModelBase(NeuralNetworkArchitecture<>,String,Int32)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputHeight` |  |
| `InputWidth` |  |
| `IsOnnxMode` |  |
| `NumClasses` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchDimension(Tensor<>)` | Adds a batch dimension to a [C, H, W] tensor, producing [1, C, H, W]. |
| `DeserializeSegmentationBaseData(BinaryReader)` | Reads common segmentation fields from a binary stream. |
| `Dispose(Boolean)` |  |
| `Forward(Tensor<>)` | Executes the full forward pass through encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass, dispatching to ONNX or native mode. |
| `PredictOnnx(Tensor<>)` | Runs ONNX inference. |
| `RemoveBatchDimension(Tensor<>)` | Removes the batch dimension from a [1, ...] tensor. |
| `Segment(Tensor<>)` |  |
| `SerializeSegmentationBaseData(BinaryWriter)` | Writes common segmentation fields to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step: forward pass, loss, backward pass, and parameter update. |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_channels` | Number of input channels (typically 3 for RGB). |
| `_disposed` | Whether this instance has been disposed. |
| `_encoderLayerEnd` | Index separating encoder layers from decoder layers in the Layers list. |
| `_height` | Input image height in pixels. |
| `_numClasses` | Number of segmentation output classes. |
| `_onnxModelPath` | Path to the ONNX model file (null in native mode). |
| `_onnxSession` | ONNX runtime inference session (null in native mode). |
| `_optimizer` | Gradient-based optimizer for training (null in ONNX mode). |
| `_useNativeMode` | Whether the model is running in native (trainable) mode or ONNX (inference-only) mode. |
| `_width` | Input image width in pixels. |

