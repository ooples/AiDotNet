---
title: "ODISESegmentation<T>"
description: "ODISE Segmentation: Panoptic segmentation via diffusion model features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Diffusion`

ODISE Segmentation: Panoptic segmentation via diffusion model features.

## For Beginners

Diffusion-based panoptic segmentation. Open-vocabulary scene understanding.

Common use cases:

- Diffusion-based panoptic segmentation
- Open-vocabulary scene understanding
- Text-guided segmentation via diffusion
- Novel category segmentation

## How It Works

**Technical Details:**

- Stable Diffusion UNet features for dense prediction
- CLIP text encoder for open-vocabulary category matching
- Mask generator trained on frozen diffusion features
- Joint text-image representation for panoptic labels

**Reference:** Xu et al., "Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ODISESegmentation(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,ODISESegmentationOptions)` | Initializes ODISESegmentation in native (trainable) mode. |
| `ODISESegmentation(NeuralNetworkArchitecture<>,String,Int32,ODISESegmentationOptions)` | Initializes ODISESegmentation in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this ODISESegmentation instance supports training. |

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

