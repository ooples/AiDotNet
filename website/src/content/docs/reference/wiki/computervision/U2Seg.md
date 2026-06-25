---
title: "U2Seg<T>"
description: "U2Seg: Unified Unsupervised Segmentation framework."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

U2Seg: Unified Unsupervised Segmentation framework.

## For Beginners

U2Seg performs instance, semantic, and panoptic segmentation without
requiring any human annotations. Instead of being trained on thousands of manually-labeled images,
U2Seg discovers object boundaries and categories through unsupervised learning — making it ideal
for domains where annotation is expensive or unavailable.

Common use cases:

- Segmentation where manual annotation is too expensive or time-consuming
- Domain adaptation (applying segmentation to new, unlabeled domains)
- Research in self-supervised and unsupervised vision
- Bootstrap labeling for subsequent supervised training

## How It Works

**Technical Details:**

- Unsupervised approach: discovers segments via self-supervised features (DINO/DINOv2)
- Uses a Mask2Former-like architecture but trained with pseudo-labels
- Handles instance, semantic, and panoptic segmentation without annotations
- Fixed architecture based on Swin-T backbone

**Reference:** Niu et al., "Unsupervised Universal Image Segmentation", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `U2Seg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,U2SegOptions)` | Initializes U2Seg in native (trainable) mode. |
| `U2Seg(NeuralNetworkArchitecture<>,String,Int32,U2SegOptions)` | Initializes U2Seg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this U2Seg instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new U2Seg instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads U2Seg configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this U2Seg model's configuration. |
| `GetOptions` | Gets the configuration options for this U2Seg model. |
| `InitializeLayers` | Initializes the encoder and decoder layers for U2Seg. |
| `PredictCore(Tensor<>)` | Runs a forward pass through U2Seg for unsupervised segmentation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes U2Seg configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step with pseudo-label supervision. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

