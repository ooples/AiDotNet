---
title: "MobileSAM<T>"
description: "MobileSAM: Faster Segment Anything with TinyViT encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Efficient`

MobileSAM: Faster Segment Anything with TinyViT encoder.

## For Beginners

Mobile segment anything. On-device interactive segmentation.

Common use cases:

- Mobile segment anything
- On-device interactive segmentation
- Lightweight promptable segmentation
- Mobile AR applications

## How It Works

**Technical Details:**

- TinyViT encoder distilled from SAM ViT-H
- Decoupled distillation: image encoder only
- Same mask decoder as original SAM
- 60x smaller encoder than SAM-ViT-H

**Reference:** Zhang et al., "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications", arXiv 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MobileSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,MobileSAMOptions)` | Initializes MobileSAM in native (trainable) mode. |
| `MobileSAM(NeuralNetworkArchitecture<>,String,Int32,MobileSAMOptions)` | Initializes MobileSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this MobileSAM instance supports training. |

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

