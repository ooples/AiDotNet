---
title: "ViTAdapter<T>"
description: "ViT-Adapter: Vision Transformer Adapter for Dense Predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Semantic`

ViT-Adapter: Vision Transformer Adapter for Dense Predictions.

## For Beginners

ViT-Adapter enables plain Vision Transformers (ViTs) to handle dense
prediction tasks like semantic segmentation without changing the ViT architecture. It adds
lightweight "adapter" modules that inject multi-scale spatial information into the ViT,
letting it produce features at different resolutions needed for pixel-level predictions.

Common use cases:

- Semantic segmentation using pre-trained ViT backbones (DINOv2, MAE, BEiT)
- Object detection with ViT encoders
- Any dense prediction task where you want to leverage powerful ViT pretraining

## How It Works

**Technical Details:**

- Spatial Prior Module: injects multi-scale spatial priors into plain ViT
- Spatial Interaction Module: enables feature interaction between adapter and ViT
- Works with any plain ViT (ViT-S, ViT-B, ViT-L) without modifying the backbone
- Adds only ~3% extra parameters on top of the base ViT
- SOTA on ADE20K with BEiT-L backbone (58.0 mIoU)

**Reference:** Chen et al., "Vision Transformer Adapter for Dense Predictions",
ICLR 2023 Spotlight.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViTAdapter(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,ViTAdapterModelSize,Double,ViTAdapterOptions)` | Initializes a new instance of ViT-Adapter in native (trainable) mode. |
| `ViTAdapter(NeuralNetworkArchitecture<>,String,Int32,ViTAdapterModelSize,ViTAdapterOptions)` | Initializes a new instance of ViT-Adapter in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this ViT-Adapter instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new ViT-Adapter with same config but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes ViT-Adapter configuration. |
| `Dispose(Boolean)` | Releases managed resources. |
| `GetModelMetadata` | Collects model metadata. |
| `GetOptions` | Gets the configuration options for this ViT-Adapter model. |
| `InitializeLayers` | Initializes the ViT + adapter encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass through the adapted ViT to produce per-pixel segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes ViT-Adapter configuration for persistence. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat vector. |

