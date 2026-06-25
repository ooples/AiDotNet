---
title: "ConsistencyDistillationSampling<T>"
description: "Consistency Distillation Sampling (CSD) for 3D generation with consistency constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Consistency Distillation Sampling (CSD) for 3D generation with consistency constraints.

## For Beginners

Regular SDS can produce 3D objects with problems like "two faces"
(Janus effect) or blurry textures. CSD fixes this by also requiring that the 3D model
looks consistent no matter how much noise is added — this extra constraint produces much
cleaner, more coherent 3D results.

## How It Works

CSD combines score distillation with consistency model training for faster and higher-quality
3D generation. It enforces that the optimized 3D representation produces consistent results
across different noise levels, reducing the multi-step dependency and artifacts common in SDS.
Produces sharper, more coherent 3D objects with fewer Janus artifacts.

Reference: Kim et al., "Consistency Trajectory Models: Learning Probability Flow ODE
Trajectory of Diffusion", ICLR 2024; adapted for 3D score distillation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConsistencyDistillationSampling(IDiffusionModel<>,Double,Double)` | Initializes a new CSD instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConsistencyWeight` | Gets the consistency loss weight. |
| `GuidanceScale` | Gets the guidance scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>,Vector<>,Vector<>,Double)` | Computes the combined SDS + consistency gradient. |

