---
title: "DRVI<T>"
description: "DRVI: disentangled representations for video interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

DRVI: disentangled representations for video interpolation.

## For Beginners

DRVI separates "what things look like" from "how things move".
By processing appearance and motion independently, it can better handle cases where
objects look similar but move differently, or where the same object appears in different
lighting conditions across frames.

**Usage:**

## How It Works

DRVI (2024) disentangles content and motion for video interpolation:

- Disentangled encoders: separate encoders for content (appearance, texture, color) and

motion (displacement, deformation) that process frame pairs independently

- Content encoder: extracts appearance features invariant to motion, shared across all

timesteps so the model doesn't re-extract appearance at each interpolation point

- Motion encoder: captures inter-frame displacement fields at multiple scales, enabling

the model to handle both global camera motion and local object motion

- Disentangled decoder: recombines content and motion representations with learned gating

at each scale, allowing fine control over which content features are warped by which
motion components

**Reference:** "DRVI: Disentangled Representations for Video Interpolation" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DRVI(NeuralNetworkArchitecture<>,DRVIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DRVI model in native training mode. |
| `DRVI(NeuralNetworkArchitecture<>,String,DRVIOptions)` | Creates a DRVI model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

