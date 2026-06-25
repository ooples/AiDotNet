---
title: "UFM<T>"
description: "UFM unified flow and matching demonstrating unified training beats specialized models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

UFM unified flow and matching demonstrating unified training beats specialized models.

## For Beginners

UFM (Unified Flow Matching) provides a unified framework for optical flow that handles both forward and backward flow estimation in a single model pass.

## How It Works

**References:**

- Paper: "UFM: Unified Flow and Matching" (2025)

UFM is the first to demonstrate that unified training for optical flow and feature matching outperforms specialized models on both tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UFM` | Creates a new UFM model for native training and inference. |

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
| `TryGetArchitectureInputShape` | UFM consumes two RGB frames concatenated channel-wise — 2 × Architecture.InputDepth = 6 channels — but Architecture.InputDepth itself reports the SINGLE-FRAME count (3) so it matches the architecture's per-frame metadata. |
| `UpdateParameters(Vector<>)` |  |

