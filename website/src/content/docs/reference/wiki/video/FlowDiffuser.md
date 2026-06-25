---
title: "FlowDiffuser<T>"
description: "FlowDiffuser diffusion-based optical flow with iterative refinement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

FlowDiffuser diffusion-based optical flow with iterative refinement.

## For Beginners

FlowDiffuser uses a diffusion process to iteratively refine optical flow estimates. Starting from random noise, it progressively denoises to produce accurate dense flow fields.

## How It Works

**References:**

- Paper: "FlowDiffuser: Advancing Optical Flow Estimation with Diffusion Models" (Luo et al., CVPR 2024)

FlowDiffuser applies diffusion models to optical flow estimation with iterative refinement, producing accurate and smooth flow fields.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowDiffuser` | Creates a new FlowDiffuser model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EstimateFlow(Tensor<>,Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

