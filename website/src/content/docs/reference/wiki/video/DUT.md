---
title: "DUT<T>"
description: "DUT deep unsupervised trajectory-based video stabilization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Stabilization`

DUT deep unsupervised trajectory-based video stabilization.

## For Beginners

DUT (Deep Unsupervised Trajectory) stabilizes shaky video by learning smooth camera trajectories without requiring ground truth stabilized footage for training.

## How It Works

**References:**

- Paper: "DUT: Learning Video Stabilization by Simply Watching Unstable Videos" (Xu et al., ICCV 2022)

DUT learns to stabilize video in an unsupervised manner by predicting per-pixel flow fields
from unstable videos alone. It uses a coarse-to-fine pyramid with temporal consistency loss
to produce smooth warping fields without requiring paired stable/unstable training data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DUT(NeuralNetworkArchitecture<>,DUTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DUT model for native training and inference. |
| `DUT(NeuralNetworkArchitecture<>,String,DUTOptions)` | Creates a DUT model for ONNX inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Stabilize(Tensor<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

