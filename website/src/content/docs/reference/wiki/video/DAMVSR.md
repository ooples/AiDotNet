---
title: "DAMVSR<T>"
description: "DAM-VSR: disentanglement of appearance and motion for video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

DAM-VSR: disentanglement of appearance and motion for video super-resolution.

## For Beginners

Most video upscalers mix up texture information with movement
information, causing blurry edges around moving objects. DAM-VSR processes them
separately - one branch focuses on making things look sharp, the other on handling
motion correctly - then combines the results intelligently.

**Usage:**

## How It Works

DAM-VSR (SIGGRAPH 2025) disentangles appearance and motion representations:

- Appearance branch: extracts texture and structural features from individual frames

using a ResNet-like encoder, capturing "what things look like"

- Motion branch: captures temporal dynamics and inter-frame correspondences using

deformable attention, learning "how things move"

- Appearance-Motion Fusion: combines both branches with learned gating, allowing the

model to balance texture detail vs motion coherence per-pixel

By separating these concerns, DAM-VSR reduces artifacts at motion boundaries
and produces sharper textures in static regions.

**Reference:** "DAM-VSR: Disentanglement of Appearance and Motion for Video
Super-Resolution" (SIGGRAPH 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DAMVSR(NeuralNetworkArchitecture<>,DAMVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DAM-VSR model in native training mode. |
| `DAMVSR(NeuralNetworkArchitecture<>,String,DAMVSROptions)` | Creates a DAM-VSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

