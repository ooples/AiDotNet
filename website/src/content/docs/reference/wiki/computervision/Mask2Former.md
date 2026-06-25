---
title: "Mask2Former<T>"
description: "Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation.

## For Beginners

Mask2Former is a universal segmentation model that handles semantic,
instance, and panoptic segmentation with a single unified architecture. Instead of designing
separate models for each task, Mask2Former uses a transformer decoder with "masked cross-attention"
that restricts each query to attend only to its predicted mask region. This makes it highly
efficient and accurate across all segmentation tasks.

Common use cases:

- Panoptic segmentation (stuff + things in one pass, e.g., road + cars + people)
- Instance segmentation (individual object masks, e.g., "car 1", "car 2")
- Semantic segmentation (per-pixel class labels, e.g., "road", "sky", "building")
- Multi-task deployment where one model serves all segmentation needs

## How It Works

**Technical Details:**

- Backbone: Swin Transformer or ResNet producing multi-scale features
- Pixel Decoder: Multi-Scale Deformable Attention Transformer (MSDeformAttn)
- Transformer Decoder: 9 layers with masked cross-attention (restricts attention to predicted masks)
- 100 learnable object queries predict class labels and binary masks
- Achieves 57.8 PQ on COCO panoptic, 83.3 AP on Cityscapes instance

**Reference:** Cheng et al., "Masked-attention Mask Transformer for Universal Image
Segmentation", CVPR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mask2Former(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,Mask2FormerModelSize,Double,Mask2FormerOptions)` | Initializes Mask2Former in native (trainable) mode. |
| `Mask2Former(NeuralNetworkArchitecture<>,String,Int32,Int32,Mask2FormerModelSize,Mask2FormerOptions)` | Initializes Mask2Former in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this Mask2Former instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new Mask2Former with same config but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Mask2Former configuration. |
| `Dispose(Boolean)` | Releases managed resources. |
| `GetModelMetadata` | Collects model metadata. |
| `GetOptions` | Gets the configuration options for this Mask2Former model. |
| `InitializeLayers` | Initializes the backbone encoder, pixel decoder, and transformer decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation masks and class predictions. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Mask2Former configuration for persistence. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step with Hungarian matching loss. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat vector. |

