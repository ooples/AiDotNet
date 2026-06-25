---
title: "RAFT<T>"
description: "Recurrent All-pairs Field Transforms (RAFT) for optical flow estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

Recurrent All-pairs Field Transforms (RAFT) for optical flow estimation.

## For Beginners

RAFT is a state-of-the-art optical flow estimation model that predicts
the motion between two consecutive video frames. Optical flow represents how pixels move
from one frame to the next, useful for:

- Motion analysis and tracking
- Video stabilization
- Action recognition
- Video compression
- Self-driving car perception

RAFT iteratively refines its flow estimate using a recurrent update mechanism,
making it very accurate while remaining efficient.

## How It Works

**Technical Details:**

- Feature extraction using CNN encoder
- 4D correlation volumes for all-pairs matching
- GRU-based iterative update operator
- Multi-scale feature pyramids

**Reference:** Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
ECCV 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RAFT` | Initializes a new instance with default architecture settings. |
| `RAFT(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,RAFTOptions)` | Initializes a new instance of the RAFT class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the input height for frames. |
| `InputWidth` | Gets the input width for frames. |
| `SupportsTraining` | Gets whether training is supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EstimateFlow(Tensor<>,Tensor<>)` | Estimates optical flow between two frames. |
| `EstimateFlowIterative(Tensor<>,Tensor<>)` | Estimates optical flow with intermediate flow predictions. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `ThrowIfNotInitialized` | Throws if the layer fields have not been initialized via `InitializeNativeLayers`. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

