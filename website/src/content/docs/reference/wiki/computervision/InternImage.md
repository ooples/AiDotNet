---
title: "InternImage<T>"
description: "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Semantic`

InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions.

## For Beginners

InternImage is a semantic segmentation model that proves CNNs can compete
with Vision Transformers when using modern deformable convolutions. It uses DCNv3 (Deformable
Convolution v3) which can adaptively adjust where it "looks" in the image based on the content,
allowing it to focus on relevant regions for better segmentation.

Common use cases:

- Large-scale scene parsing (Cityscapes, ADE20K)
- Object detection and segmentation pipelines
- Foundation model applications requiring dense predictions

## How It Works

**Technical Details:**

- DCNv3 operator with multi-group deformable attention
- 4-stage hierarchical architecture (like ConvNeXt/Swin but with DCNv3)
- UPerNet decoder for multi-scale feature aggregation
- Scales from 30M (Tiny) to 1.08B (Huge) parameters
- Competitive with ViT-based models on ADE20K and COCO

**Reference:** Wang et al., "InternImage: Exploring Large-Scale Vision Foundation Models
with Deformable Convolutions", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InternImage(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,InternImageModelSize,Double,InternImageOptions)` | Initializes a new instance of InternImage in native (trainable) mode. |
| `InternImage(NeuralNetworkArchitecture<>,String,Int32,InternImageModelSize,InternImageOptions)` | Initializes a new instance of InternImage in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this InternImage instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new InternImage with the same config but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes InternImage-specific configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX session. |
| `GetModelMetadata` | Collects metadata describing this InternImage model. |
| `GetOptions` | Gets the configuration options for this InternImage model. |
| `InitializeLayers` | Initializes the DCNv3 encoder and UPerNet decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass through InternImage to produce per-pixel segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes InternImage-specific configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step with forward pass, loss computation, backward pass, and update. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat vector. |

