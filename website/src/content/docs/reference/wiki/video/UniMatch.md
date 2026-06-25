---
title: "UniMatch<T>"
description: "UniMatch unified flow, stereo, and depth estimation with cross-task transfer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

UniMatch unified flow, stereo, and depth estimation with cross-task transfer.

## For Beginners

UniMatch is a unified model for dense matching that handles optical flow, stereo matching, and depth estimation with a single architecture. It learns general correspondence features.

## How It Works

**References:**

- Paper: "Unifying Flow, Stereo and Depth Estimation" (Xu et al., TPAMI 2023)

UniMatch unifies optical flow, stereo matching, and depth estimation in a single architecture, enabling cross-task transfer learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniMatch` | Initializes a new instance with default architecture settings. |
| `UniMatch(NeuralNetworkArchitecture<>,Int32,Int32,UniMatchOptions)` | Creates a new UniMatch model for native training and inference. |

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

