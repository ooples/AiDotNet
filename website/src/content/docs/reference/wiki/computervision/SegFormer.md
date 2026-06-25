---
title: "SegFormer<T>"
description: "SegFormer: Simple and Efficient Semantic Segmentation with Transformers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Semantic`

SegFormer: Simple and Efficient Semantic Segmentation with Transformers.

## For Beginners

SegFormer is a semantic segmentation model that classifies every pixel
in an image into a category (e.g., road, sky, person, building). It uses a hierarchical
transformer encoder (Mix Transformer / MiT) and a lightweight MLP decoder — no complex
decoders or positional encodings needed.

Common use cases:

- Autonomous driving (road, lane, obstacle detection)
- Medical imaging (organ/lesion segmentation)
- Scene understanding (indoor/outdoor parsing)
- Agriculture (crop/weed detection from drone images)

## How It Works

**Technical Details:**

- 4-stage hierarchical encoder producing multi-scale features (1/4, 1/8, 1/16, 1/32)
- Overlapping patch embedding replaces non-overlapping ViT patches
- Efficient Self-Attention with spatial reduction for computational savings
- Mix-FFN uses 3x3 depthwise convolutions instead of positional encodings
- Lightweight All-MLP decode head for fast inference
- Six model sizes (B0–B5) from 3.8M to 82.0M parameters

**Reference:** Xie et al., "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers", NeurIPS 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegFormer(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SegFormerModelSize,Double,SegFormerOptions)` | Initializes a new instance of SegFormer in native (trainable) mode. |
| `SegFormer(NeuralNetworkArchitecture<>,String,Int32,SegFormerModelSize,SegFormerOptions)` | Initializes a new instance of SegFormer in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelSize` | Gets the model size variant (B0 through B5). |
| `NumClasses` | Gets the number of semantic classes this model predicts. |
| `SupportsTraining` | Gets whether this SegFormer instance supports training. |
| `UseNativeMode` | Gets whether using native mode (trainable) or ONNX mode (inference only). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchDimension(Tensor<>)` | Adds a batch dimension to an unbatched [C, H, W] tensor, producing [1, C, H, W]. |
| `CreateNewInstance` | Creates a new SegFormer instance with the same configuration as this one but freshly initialized weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads SegFormer-specific configuration values from a binary stream during model loading. |
| `Dispose(Boolean)` | Releases managed resources held by this SegFormer instance, including the ONNX inference session. |
| `Forward(Tensor<>)` | Executes the full forward pass through the SegFormer encoder and decoder in native mode. |
| `GetModelConfig(SegFormerModelSize)` | Returns the architecture configuration (embedding dimensions, transformer depths, attention head counts, and decoder dimension) for a given SegFormer model size. |
| `GetModelMetadata` | Collects metadata describing this SegFormer model's configuration and current state. |
| `GetOptions` | Gets the configuration options for this SegFormer model. |
| `InitializeLayers` | Initializes the encoder and decoder layers for the SegFormer model. |
| `PredictCore(Tensor<>)` | Runs a forward pass through the SegFormer model to produce a per-pixel segmentation map. |
| `PredictOnnx(Tensor<>)` | Runs inference using the ONNX runtime session for optimized prediction. |
| `RemoveBatchDimension(Tensor<>)` | Removes the batch dimension from a [1, C, H, W] tensor, producing [C, H, W]. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes all SegFormer-specific configuration values to a binary stream for model persistence. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step: forward pass, loss computation, backward pass, and parameter update. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters across the encoder and decoder layers from a flat parameter vector. |

