---
title: "ModalityType"
description: "Identifies the type of data modality in a multimodal sample."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Multimodal`

Identifies the type of data modality in a multimodal sample.

## For Beginners

A "modality" means a type of data. Modern AI models
often work with multiple types at once - for example, an image captioning model
uses both images and text. This enum identifies which type each piece of data is.

## Fields

| Field | Summary |
|:-----|:--------|
| `Audio` | Audio data (waveform samples or spectrograms). |
| `Custom` | Custom or unspecified modality type. |
| `Graph` | Graph-structured data (nodes and edges). |
| `Image` | Image data (2D pixel grids, typically [H, W, C] or [C, H, W]). |
| `PointCloud` | 3D point cloud data (sets of 3D coordinates with optional features). |
| `Tabular` | Structured/tabular data (feature vectors). |
| `Text` | Text data (token sequences, embeddings, or raw strings). |
| `TimeSeries` | Time series data (sequential measurements over time). |
| `Video` | Video data (sequences of frames, typically [T, H, W, C]). |

