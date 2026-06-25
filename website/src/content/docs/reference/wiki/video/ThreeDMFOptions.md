---
title: "ThreeDMFOptions"
description: "Configuration options for the 3DMF (3D Motion Field) video stabilization model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the 3DMF (3D Motion Field) video stabilization model.

## For Beginners

3DMF options configure the 3D motion field video stabilizer.

## How It Works

**References:**

- Paper: "3D Video Stabilization with Depth Estimation by CNN-based Optimization" (Lee & Lee, CVPR 2021)

3DMF estimates depth and 3D camera motion to perform stabilization in 3D space,
better handling parallax and depth-dependent motion than 2D methods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThreeDMFOptions` | Initializes a new instance with default values. |
| `ThreeDMFOptions(ThreeDMFOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumDepthLayers` | Number of depth estimation layers in the depth prediction branch. |
| `NumFeatures` | Number of base feature channels. |
| `NumMotionIters` | Number of 3D motion estimation refinement iterations. |
| `NumResBlocks` | Number of residual blocks in the feature backbone. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

