---
title: "EfficientSAM<T>"
description: "EfficientSAM: Leveraged Masked Image Pretraining for efficient SAM."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Efficient`

EfficientSAM: Leveraged Masked Image Pretraining for efficient SAM.

## For Beginners

Efficient segment anything with strong pre-training. Interactive segmentation.

Common use cases:

- Efficient segment anything with strong pre-training
- Interactive segmentation
- Automatic mask generation
- Production SAM deployment

## How It Works

**Technical Details:**

- SAMI (SAM Image) pre-training with masked autoencoder
- Lightweight ViT encoder with MAE knowledge transfer
- Cross-attention decoder for mask prediction
- Better quality-efficiency trade-off than MobileSAM

**Reference:** Xiong et al., "EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EfficientSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,EfficientSAMOptions)` | Initializes EfficientSAM in native (trainable) mode. |
| `EfficientSAM(NeuralNetworkArchitecture<>,String,Int32,EfficientSAMOptions)` | Initializes EfficientSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this EfficientSAM instance supports training. |

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

