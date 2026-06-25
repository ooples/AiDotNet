---
title: "DepthAnythingV2<T>"
description: "Depth Anything V2 for monocular depth estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Depth`

Depth Anything V2 for monocular depth estimation.

## For Beginners

Depth Anything V2 is a state-of-the-art model for estimating depth maps
from single images (monocular depth estimation). Given an RGB image, it predicts the relative
distance of each pixel from the camera. This is useful for:

- 3D scene understanding
- Augmented reality applications
- Autonomous driving
- Video editing and VFX
- Object detection and segmentation

Unlike stereo depth estimation which requires two cameras, Depth Anything works with
a single image by learning depth cues from large-scale training data.

## How It Works

**Technical Details:**

- Vision Transformer (ViT) based encoder with DINOv2 initialization
- Efficient multi-scale decoder for dense prediction
- Scale-invariant depth loss for robust training
- Supports various backbone sizes (Small, Base, Large)

**Reference:** Yang et al., "Depth Anything V2" 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DepthAnythingV2` | Initializes a new instance of the DepthAnythingV2 class in native (trainable) mode. |
| `DepthAnythingV2(NeuralNetworkArchitecture<>,String,DepthAnythingV2<>.ModelSize,DepthAnythingV2Options)` | Initializes a new instance of the DepthAnythingV2 class in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the input height for frames. |
| `InputWidth` | Gets the input width for frames. |
| `Size` | Gets the model size variant. |
| `SupportsTraining` | Gets whether training is supported. |
| `UseNativeMode` | Gets whether using native mode (trainable) or ONNX mode (inference only). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` | Disposes of managed resources, including the ONNX inference session. |
| `EstimateDepth(Tensor<>)` | Estimates depth from an RGB image. |
| `EstimateVideoDepth(List<Tensor<>>)` | Estimates depth for a sequence of video frames. |
| `GetDepthAtPoint(Tensor<>,Int32,Int32)` | Gets the relative depth value at a specific point. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `PredictOnnx(Tensor<>)` | Performs ONNX inference for depth estimation. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

