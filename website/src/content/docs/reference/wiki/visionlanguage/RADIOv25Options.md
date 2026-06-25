---
title: "RADIOv25Options"
description: "Configuration options for RADIOv2.5, NVIDIA's agglomerative vision foundation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for RADIOv2.5, NVIDIA's agglomerative vision foundation model.

## How It Works

RADIOv2.5 (Ranzinger et al., 2025) distills multiple teacher models (DINOv2, SAM, SigLIP, CLIP)
into a single student model via multi-teacher distillation. The resulting model produces features
compatible with all teacher models, serving as a universal vision backbone.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RADIOv25Options(RADIOv25Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterDim` | Gets or sets the adapter dimension for teacher-specific heads. |
| `NumSummaryTokens` | Gets or sets the number of summary tokens for compact representation. |
| `TeacherModels` | Gets or sets the teacher models used for distillation. |
| `UseTeacherSpecificHeads` | Gets or sets whether to produce teacher-specific output heads. |

