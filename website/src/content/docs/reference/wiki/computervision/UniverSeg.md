---
title: "UniverSeg<T>"
description: "UniverSeg: Universal Medical Image Segmentation via cross-attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

UniverSeg: Universal Medical Image Segmentation via cross-attention.

## For Beginners

Few-shot medical segmentation without fine-tuning. Cross-domain medical image segmentation.

Common use cases:

- Few-shot medical segmentation without fine-tuning
- Cross-domain medical image segmentation
- Label-efficient medical AI
- New task adaptation from a few examples

## How It Works

**Technical Details:**

- CrossBlock mechanism for support-query feature interaction
- No fine-tuning needed for new segmentation tasks
- Uses a small labeled support set at inference time
- Trained on diverse medical segmentation datasets

**Reference:** Butoi et al., "UniverSeg: Universal Medical Image Segmentation", ICCV 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniverSeg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,UniverSegOptions)` | Initializes UniverSeg in native (trainable) mode. |
| `UniverSeg(NeuralNetworkArchitecture<>,String,Int32,UniverSegOptions)` | Initializes UniverSeg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this UniverSeg instance supports training. |

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

