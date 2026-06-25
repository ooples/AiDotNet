---
title: "GIMMVFIOptions"
description: "Configuration options for the GIMM-VFI generalizable implicit motion modeling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the GIMM-VFI generalizable implicit motion modeling.

## For Beginners

GIMM-VFI learns a smooth, continuous "motion field" that describes
how everything in the scene moves over time. Once it processes two frames, it can generate
a new frame at ANY point in time between them (not just the midpoint). This is great for
creating variable slow-motion effects or non-uniform frame rate conversion.

## How It Works

GIMM-VFI (NeurIPS 2024) uses implicit neural representations for continuous-time motion:

- Implicit motion function: learns a continuous function M(x, y, t) that maps any spatial

position (x, y) and any timestep t in [0, 1] to a motion vector, enabling interpolation
at arbitrary (non-uniform) time intervals without retraining

- Motion encoding network: encodes the two input frames into a shared motion latent space

using cross-correlation features, which the implicit function queries to produce per-pixel
motion at any desired timestep

- Generalizable across timesteps: a single forward pass of the motion encoder produces a

representation that the implicit function can query at any t, unlike methods that need
separate inference per timestep

- Adaptive sampling: the implicit function can be queried at higher density in regions with

complex motion and lower density in static regions for efficient computation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GIMMVFIOptions` | Initializes a new instance with default values. |
| `GIMMVFIOptions(GIMMVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `ImplicitDim` | Gets or sets the dimension of the implicit motion representation. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumEncoderBlocks` | Gets or sets the number of motion encoder residual blocks. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFrequencies` | Gets or sets the number of positional encoding frequencies. |
| `NumImplicitLayers` | Gets or sets the number of MLP layers in the implicit function. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

