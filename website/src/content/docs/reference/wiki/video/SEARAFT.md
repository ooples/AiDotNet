---
title: "SEARAFT<T>"
description: "SEA-RAFT simple efficient accurate RAFT with mixture of Laplace loss."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

SEA-RAFT simple efficient accurate RAFT with mixture of Laplace loss.

## For Beginners

SEA-RAFT extends RAFT with Scale-Equivariant Architecture for better handling of objects at different scales. It improves flow accuracy for both small and large motions.

## How It Works

**References:**

- Paper: "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow" (Teed et al., ECCV 2024 Oral, Best Paper Candidate)

SEA-RAFT simplifies RAFT with a mixture of Laplace loss and direct initial flow prediction, achieving state-of-the-art accuracy with improved efficiency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SEARAFT` | Creates a new SEARAFT model for native training and inference. |

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

