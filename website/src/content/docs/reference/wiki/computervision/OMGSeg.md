---
title: "OMGSeg<T>"
description: "OMG-Seg: Is One Model Good Enough For All Segmentation?"
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

OMG-Seg: Is One Model Good Enough For All Segmentation?

## For Beginners

OMG-Seg answers "yes" — one model handles over 10 segmentation tasks with
only 70M trainable parameters. Instead of training separate models for semantic, instance, panoptic,
video, interactive, and open-vocabulary segmentation, OMG-Seg uses task-specific queries to switch
between tasks at inference time.

Common use cases:

- Multi-task segmentation systems needing many task types
- Resource-constrained environments where one model must serve all needs
- Research comparing different segmentation paradigms
- Production systems with diverse segmentation requirements

## How It Works

**Technical Details:**

- Shared transformer backbone with task-specific query embeddings
- Each task type has its own set of queries that specialize during training
- Supports image (semantic, instance, panoptic) and video segmentation
- Open-vocabulary capability through text-conditioned queries
- Only 70M trainable parameters with frozen backbone

**Reference:** Li et al., "OMG-Seg: Is One Model Good Enough For All Segmentation?", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OMGSeg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,OMGSegModelSize,Double,OMGSegOptions)` | Initializes OMG-Seg in native (trainable) mode. |
| `OMGSeg(NeuralNetworkArchitecture<>,String,Int32,Int32,OMGSegModelSize,OMGSegOptions)` | Initializes OMG-Seg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this OMG-Seg instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new OMG-Seg instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads OMG-Seg configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this OMG-Seg model's configuration. |
| `GetOptions` | Gets the configuration options for this OMG-Seg model. |
| `InitializeLayers` | Initializes the encoder and decoder layers for OMG-Seg. |
| `PredictCore(Tensor<>)` | Runs a forward pass through OMG-Seg for multi-task segmentation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes OMG-Seg configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

