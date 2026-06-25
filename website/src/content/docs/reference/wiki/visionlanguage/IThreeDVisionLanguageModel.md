---
title: "IThreeDVisionLanguageModel<T>"
description: "Interface for 3D vision-language models that understand point clouds, 3D scenes, and spatial relationships."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for 3D vision-language models that understand point clouds, 3D scenes, and spatial relationships.

## How It Works

3D vision-language models extend traditional 2D VLMs to process point clouds, voxel grids,
and 3D scene representations. They enable spatial reasoning, 3D object grounding,
and scene-level question answering.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxPoints` | Gets the maximum number of 3D points the model can process. |
| `PointChannels` | Gets the number of channels per point (e.g., 3 for XYZ, 6 for XYZ+RGB). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFrom3D(Tensor<>,String)` | Processes a 3D point cloud and generates language output conditioned on a text prompt. |

