---
title: "OneFormer<T>"
description: "OneFormer: One Transformer to Rule Universal Image Segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

OneFormer: One Transformer to Rule Universal Image Segmentation.

## For Beginners

OneFormer is trained once on panoptic data and can then perform any
segmentation task — semantic, instance, or panoptic — by simply providing a text prompt that
describes which task to perform. This "one model for all tasks" approach dramatically simplifies
deployment compared to maintaining separate models for each task.

Example usage:

- Pass "the task is semantic" to get per-pixel class labels
- Pass "the task is instance" to get individual object masks
- Pass "the task is panoptic" to get both stuff and thing segments

Common use cases:

- Multi-task segmentation systems needing all three task types
- Research comparing segmentation approaches
- Production systems where maintaining one model is simpler than three

## How It Works

**Technical Details:**

- Builds on Mask2Former with a text encoder (CLIP-based) for task conditioning
- Task-conditioned joint training on panoptic, semantic, and instance data simultaneously
- Uses a task-guided query initialization that focuses queries on the specified task
- Backbone: Swin-L or DiNAT-L (Dilated Neighborhood Attention Transformer)
- SOTA on ADE20K, Cityscapes, and COCO across all three tasks with a single model

**Reference:** Jain et al., "OneFormer: One Transformer to Rule Universal Image
Segmentation", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneFormer(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,OneFormerModelSize,Double,OneFormerOptions)` | Initializes OneFormer in native (trainable) mode. |
| `OneFormer(NeuralNetworkArchitecture<>,String,Int32,Int32,OneFormerModelSize,OneFormerOptions)` | Initializes OneFormer in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this OneFormer instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new OneFormer with same config but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes OneFormer configuration. |
| `Dispose(Boolean)` | Releases managed resources. |
| `GetModelMetadata` | Collects model metadata. |
| `GetOptions` | Gets the configuration options for this OneFormer model. |
| `InitializeLayers` | Initializes the backbone encoder, text encoder, and transformer decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass through OneFormer for task-conditioned segmentation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes OneFormer configuration. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step with panoptic multi-task learning. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat vector. |

