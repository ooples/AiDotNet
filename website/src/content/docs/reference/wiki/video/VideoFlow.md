---
title: "VideoFlow<T>"
description: "VideoFlow multi-frame temporal cues for optical flow estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

VideoFlow multi-frame temporal cues for optical flow estimation.

## For Beginners

VideoFlow extends optical flow estimation to video sequences by exploiting temporal continuity. It produces smoother, more consistent flow fields across consecutive frames.

## How It Works

**References:**

- Paper: "VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation" (Shi et al., 2023)

VideoFlow exploits temporal cues from multiple frames simultaneously to improve optical flow estimation accuracy and temporal consistency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoFlow` | Initializes a new instance with default architecture settings. |
| `VideoFlow(NeuralNetworkArchitecture<>,Int32,Int32,VideoFlowOptions)` | Creates a new VideoFlow model for native training and inference. |

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

