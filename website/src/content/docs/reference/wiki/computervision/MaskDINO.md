---
title: "MaskDINO<T>"
description: "Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation.

## For Beginners

Mask DINO extends the powerful DINO object detector with a mask prediction
branch, creating a unified architecture that handles object detection, instance segmentation,
panoptic segmentation, and semantic segmentation all in one model. Instead of building separate
models for each task, Mask DINO uses a shared backbone and query-based transformer to do everything.

Common use cases:

- Joint object detection + instance segmentation
- Panoptic segmentation (detecting all things and stuff)
- Research requiring a unified detection-segmentation framework
- Production systems needing both detection boxes and segmentation masks

## How It Works

**Technical Details:**

- Built on DINO detector with deformable attention transformer encoder-decoder
- Adds a mask branch using dot product between query embeddings and pixel embeddings
- Unified query matching for both box and mask predictions via Hungarian matching
- Backbone: ResNet-50 or Swin-L
- Achieves 54.5 AP on COCO instance, 59.4 PQ on COCO panoptic

**Reference:** Li et al., "Mask DINO: Towards A Unified Transformer-based Framework
for Object Detection and Segmentation", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskDINO(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,MaskDINOModelSize,Double,MaskDINOOptions)` | Initializes Mask DINO in native (trainable) mode. |
| `MaskDINO(NeuralNetworkArchitecture<>,String,Int32,Int32,MaskDINOModelSize,MaskDINOOptions)` | Initializes Mask DINO in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this Mask DINO instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new Mask DINO instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads Mask DINO configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this Mask DINO model's configuration. |
| `GetOptions` | Gets the configuration options for this Mask DINO model. |
| `InitializeLayers` | Initializes the encoder and decoder layers for Mask DINO. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce detection boxes and segmentation masks. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes Mask DINO configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step with forward, loss, backward, and parameter update. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

