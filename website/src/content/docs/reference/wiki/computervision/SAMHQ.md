---
title: "SAMHQ<T>"
description: "SAM-HQ: Segment Anything in High Quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

SAM-HQ: Segment Anything in High Quality.

## For Beginners

SAM-HQ extends Meta's Segment Anything Model with a High-Quality output
token that produces significantly sharper and more accurate mask boundaries. While SAM sometimes
produces coarse masks (especially on thin structures like bicycle spokes or fences), SAM-HQ adds
learnable components that refine boundaries to pixel-level precision.

Common use cases:

- High-precision object segmentation where boundary quality matters
- Thin structure segmentation (wires, fences, poles)
- Medical imaging requiring precise boundaries
- Any SAM use case where mask quality needs improvement

## How It Works

**Technical Details:**

- Adds an HQ output token alongside SAM's original output tokens
- Global-local feature fusion: combines early ViT features (local) with final features (global)
- Trained on only 44K fine-grained masks from HQSeg-44K dataset
- +17.6 mBIoU improvement on DIS-val5K (thin/complex structures)
- Backbone: ViT-B/L/H (same as SAM)

**Reference:** Ke et al., "Segment Anything in High Quality", NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAMHQ(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SAMHQModelSize,Double,SAMHQOptions)` | Initializes SAM-HQ in native (trainable) mode. |
| `SAMHQ(NeuralNetworkArchitecture<>,String,Int32,SAMHQModelSize,SAMHQOptions)` | Initializes SAM-HQ in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SAM-HQ instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new SAM-HQ instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads SAM-HQ configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this SAM-HQ model's configuration. |
| `GetOptions` | Gets the configuration options for this SAM-HQ model. |
| `InitializeLayers` | Initializes the encoder and decoder layers for SAM-HQ. |
| `PredictCore(Tensor<>)` | Runs a forward pass through SAM-HQ to produce high-quality segmentation masks. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes SAM-HQ configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step: forward pass, loss computation, backward pass, and parameter update. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

