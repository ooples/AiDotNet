---
title: "GPT4Point<T>"
description: "GPT4Point: unified point-language understanding and generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.ThreeD`

GPT4Point: unified point-language understanding and generation.

## For Beginners

GPT4Point is a vision-language model for understanding and
generating 3D point clouds using natural language. Default values follow the original
paper settings.

## How It Works

GPT4Point (2024) is a unified framework for point cloud-language understanding and generation.
It bridges 3D point cloud representations with language models through a point cloud encoder
that projects geometric features into the LLM's embedding space, enabling both point cloud
captioning and text-conditioned 3D point cloud generation in a single model.

**References:**

- Paper: "GPT4Point: A Unified Framework for Point-Language Understanding and Generation (Various, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFrom3D(Tensor<>,String)` | Processes 3D point cloud using GPT4Point's Point-QFormer alignment approach. |
| `GenerateFromImage(Tensor<>,String)` | Generates from 2D image using GPT4Point's Point-QFormer alignment. |

