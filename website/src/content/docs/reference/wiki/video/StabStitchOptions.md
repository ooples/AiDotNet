---
title: "StabStitchOptions"
description: "Configuration options for the StabStitch video stabilization model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the StabStitch video stabilization model.

## For Beginners

StabStitch options configure the joint stabilization and stitching model.

## How It Works

**References:**

- Paper: "StabStitch: Simultaneous Video Stabilization and Stitching" (2023)

StabStitch jointly performs video stabilization and stitching, simultaneously removing
camera shake while producing a seamless panoramic output from moving cameras.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StabStitchOptions` | Initializes a new instance with default values. |
| `StabStitchOptions(StabStitchOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `MeshGridCols` | Number of mesh grid columns for thin-plate-spline warping. |
| `MeshGridRows` | Number of mesh grid rows for thin-plate-spline warping. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels. |
| `NumWarpBranches` | Number of warp estimation branches (one per camera stream). |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

