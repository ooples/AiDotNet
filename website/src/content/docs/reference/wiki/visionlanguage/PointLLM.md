---
title: "PointLLM<T>"
description: "PointLLM: LLM understanding of colored 3D point clouds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.ThreeD`

PointLLM: LLM understanding of colored 3D point clouds.

## For Beginners

PointLLM is a vision-language model that enables LLMs to
understand and reason about 3D point cloud data. Default values follow the original
paper settings.

## How It Works

PointLLM (OpenRobot Lab, 2024) empowers large language models to understand colored 3D
point clouds. It encodes point cloud spatial coordinates and color features through a
point cloud backbone, projects them into the LLM's token space, and enables the model
to reason about 3D object shapes, colors, and spatial relationships through natural language.

**References:**

- Paper: "PointLLM: Empowering Large Language Models to Understand Point Clouds (OpenRobot Lab, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFrom3D(Tensor<>,String)` | Processes 3D point cloud using PointLLM's point cloud tokenization approach. |
| `GenerateFromImage(Tensor<>,String)` | Generates from 2D image using PointLLM's point cloud tokenization approach. |

