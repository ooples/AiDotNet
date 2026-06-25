---
title: "SAM<T>"
description: "Segment Anything Model (SAM): the first promptable foundation model for image segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

Segment Anything Model (SAM): the first promptable foundation model for image segmentation.

## For Beginners

SAM can segment any object in any image given a prompt (point click,
bounding box, or text). It was trained on the SA-1B dataset containing over 1 billion masks
across 11 million images, making it extremely versatile.

Common use cases:

- Interactive object selection in image editors
- Automatic mask generation for datasets
- Zero-shot transfer to new domains without fine-tuning
- Foundation for downstream segmentation tasks

## How It Works

**Technical Details:**

- ViT-H/L/B image encoder with 16x16 patch embedding
- Lightweight prompt encoder for points, boxes, and masks
- Two-way transformer mask decoder with IoU prediction head
- Ambiguity-aware: predicts 3 masks per prompt (whole, part, subpart)
- 1024x1024 input resolution; encoder runs once per image

**Reference:** Kirillov et al., "Segment Anything", ICCV 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SAMModelSize,Double,SAMOptions)` | Initializes SAM in native (trainable) mode. |
| `SAM(NeuralNetworkArchitecture<>,String,Int32,SAMModelSize,SAMOptions)` | Initializes SAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SAM instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources. |
| `GetModelMetadata` | Collects metadata describing this SAM model's configuration. |
| `GetOptions` | Gets the configuration options for this SAM model. |
| `InitializeLayers` | Initializes the ViT encoder and mask decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass through SAM to produce segmentation mask logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

