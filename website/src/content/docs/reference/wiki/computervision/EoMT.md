---
title: "EoMT<T>"
description: "EoMT: Encoder-only Mask Transformer for universal image segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

EoMT: Encoder-only Mask Transformer for universal image segmentation.

## For Beginners

EoMT dramatically simplifies segmentation by removing the complex pixel
decoder and transformer decoder used by models like Mask2Former. Instead, mask queries are
inserted directly into a plain Vision Transformer (DINOv2), making EoMT 4.4x faster while
maintaining competitive accuracy. Think of it as the "minimalist" approach to segmentation.

Common use cases:

- Real-time panoptic segmentation
- Latency-sensitive deployment (4.4x faster than Mask2Former)
- Research into simpler segmentation architectures
- Any scenario where speed matters more than peak accuracy

## How It Works

**Technical Details:**

- Uses DINOv2 as frozen backbone (ViT-S/B/L)
- Queries inserted at intermediate ViT layers, processed alongside image tokens
- No separate pixel decoder or transformer decoder needed
- Query-to-mask via dot product with intermediate ViT features
- 4.4x faster than Mask2Former-Swin-L with competitive results

**Reference:** Saporta et al., "Encoder-only Mask Transformer", CVPR 2025 Highlight.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EoMT(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,EoMTModelSize,Double,EoMTOptions)` | Initializes EoMT in native (trainable) mode. |
| `EoMT(NeuralNetworkArchitecture<>,String,Int32,Int32,EoMTModelSize,EoMTOptions)` | Initializes EoMT in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this EoMT instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new EoMT instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads EoMT configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this EoMT model's configuration. |
| `GetOptions` | Gets the configuration options for this EoMT model. |
| `InitializeLayers` | Initializes the encoder-only layers for EoMT. |
| `PredictCore(Tensor<>)` | Runs a forward pass through EoMT for efficient segmentation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes EoMT configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

