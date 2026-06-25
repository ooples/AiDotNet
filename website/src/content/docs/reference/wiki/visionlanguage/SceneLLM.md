---
title: "SceneLLM<T>"
description: "Scene-LLM: voxel-based 3D scene understanding with language models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.ThreeD`

Scene-LLM: voxel-based 3D scene understanding with language models.

## For Beginners

Scene-LLM is a vision-language model for voxel-based 3D scene
understanding with language-guided spatial reasoning. Default values follow the original
paper settings.

## How It Works

Scene-LLM (2024) extends language models for 3D visual understanding and reasoning using
voxel-based scene representations. It discretizes 3D environments into voxel grids with
semantic features, enabling language-guided spatial reasoning about object locations,
room layouts, and navigation instructions in indoor environments.

**References:**

- Paper: "Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning (Various, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFrom3D(Tensor<>,String)` | Processes 3D point cloud using SceneLLM's hybrid scene representation. |
| `GenerateFromImage(Tensor<>,String)` | Generates from 2D image using SceneLLM's hybrid scene representation. |

