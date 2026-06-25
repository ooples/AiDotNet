---
title: "NeckBase<T>"
description: "Base class for neck modules that perform multi-scale feature fusion."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Detection.Necks`

Base class for neck modules that perform multi-scale feature fusion.

## For Beginners

The "neck" sits between the backbone and the detection head.
It takes multi-scale features from the backbone and fuses them together so that
each feature level contains information from both higher and lower resolutions.
This helps detect objects of various sizes more accurately.

## How It Works

Common neck architectures:

- FPN (Feature Pyramid Network): Top-down feature fusion
- PANet (Path Aggregation Network): Top-down + bottom-up paths
- BiFPN (Bidirectional FPN): Weighted bidirectional fusion

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeckBase` | Creates a new neck module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Name` | Name of this neck architecture. |
| `NumLevels` | Number of feature levels output by the neck. |
| `OutputChannels` | Number of output channels for all feature levels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(Tensor<>,Tensor<>)` | Adds two feature maps element-wise. |
| `Conv1x1(Tensor<>,Tensor<>,Tensor<>)` | Applies a 1x1 convolution to change the number of channels. |
| `DeepCopy` | Concrete necks are responsible for producing a true deep copy of their internal Conv2D wrappers and config. |
| `Downsample2x(Tensor<>)` | Downsample a feature map by a factor of 2 using max pooling. |
| `Forward(List<Tensor<>>)` | Performs multi-scale feature fusion. |
| `GetParameterCount` | Gets the total number of parameters in the neck. |
| `GetParameters` | Neck parameters live inside per-stage Conv2D wrappers and are serialized via `WriteParameters`/`ReadParameters` on the concrete neck subclass. |
| `Predict(Tensor<>)` | Single-tensor `Predict` is not a meaningful operation for a detection neck: concrete necks (FPN, PANet, BiFPN) operate on the full backbone feature pyramid (a `List` with one tensor per level) and would fail their own feature-count validati… |
| `ReadParameters(BinaryReader)` | Reads parameters from a binary reader for deserialization. |
| `SetParameters(Vector<>)` | See `GetParameters`. |
| `SetTrainingMode(Boolean)` | Sets whether the neck is in training mode. |
| `Train(Tensor<>,Tensor<>)` | Detection necks (FPN, PANet, BiFPN) are not standalone-trainable: they are trained as part of a parent detector that orchestrates the joint backbone+neck+head pass. |
| `Upsample2x(Tensor<>)` | Upsample a feature map by a factor of 2 using nearest neighbor interpolation. |
| `ValidateFeatures(List<Tensor<>>,Int32[])` | Validates that the input features are compatible with this neck. |
| `WithParameters(Vector<>)` | See `GetParameters`. |
| `WriteParameters(BinaryWriter)` | Writes all parameters to a binary writer for serialization. |

## Fields

| Field | Summary |
|:-----|:--------|
| `IsTrainingMode` | Whether the neck is in training mode. |

