---
title: "KMaXDeepLab<T>"
description: "kMaX-DeepLab: k-means Mask Transformer for panoptic segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Panoptic`

kMaX-DeepLab: k-means Mask Transformer for panoptic segmentation.

## For Beginners

Panoptic segmentation (stuff + things). Scene understanding for autonomous driving.

Common use cases:

- Panoptic segmentation (stuff + things)
- Scene understanding for autonomous driving
- Robotics environment parsing
- Dense scene labeling

## How It Works

**Technical Details:**

- k-means cross-attention replacing standard cross-attention
- Cluster centers as mask queries updated iteratively
- Pixel-path and cluster-path dual decoder
- State-of-the-art panoptic quality on COCO and Cityscapes

**Reference:** Yu et al., "k-means Mask Transformer", ECCV 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KMaXDeepLab(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,KMaXDeepLabModelSize,Double,KMaXDeepLabOptions)` | Initializes KMaXDeepLab in native (trainable) mode. |
| `KMaXDeepLab(NeuralNetworkArchitecture<>,String,Int32,KMaXDeepLabModelSize,KMaXDeepLabOptions)` | Initializes KMaXDeepLab in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this KMaXDeepLab instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `InitializeLayers` | Initializes the encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

