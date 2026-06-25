---
title: "SegNeXt<T>"
description: "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Semantic`

SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation.

## For Beginners

SegNeXt is a semantic segmentation model that labels every pixel
in an image with a category (e.g., road, tree, building). Unlike transformer-based models,
SegNeXt uses a purely convolutional backbone called MSCAN (Multi-Scale Convolutional Attention
Network) that achieves better accuracy than many transformers while being simpler and faster.

Common use cases:

- Autonomous driving (road, lane, sidewalk parsing)
- Drone/satellite imagery analysis
- Indoor scene understanding
- Real-time segmentation on resource-constrained devices

## How It Works

**Technical Details:**

- MSCAN backbone with multi-scale convolutional attention (no self-attention needed)
- Multi-branch depth-wise strip convolutions capture multi-scale context
- Attention weights are computed via 1x1 convolutions on concatenated multi-scale features
- Hamburger decoder uses matrix decomposition for global context aggregation
- Four model sizes (Tiny to Large) from 4.3M to 48.9M parameters

**Reference:** Guo et al., "SegNeXt: Rethinking Convolutional Attention Design for
Semantic Segmentation", NeurIPS 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegNeXt(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SegNeXtModelSize,Double,SegNeXtOptions)` | Initializes a new instance of SegNeXt in native (trainable) mode. |
| `SegNeXt(NeuralNetworkArchitecture<>,String,Int32,SegNeXtModelSize,SegNeXtOptions)` | Initializes a new instance of SegNeXt in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelSize` | Gets the model size variant (Tiny through Large). |
| `NumClasses` | Gets the number of semantic classes this model predicts. |
| `SupportsTraining` | Gets whether this SegNeXt instance supports training. |
| `UseNativeMode` | Gets whether using native mode (trainable) or ONNX mode (inference only). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchDimension(Tensor<>)` | Adds a batch dimension to an unbatched [C, H, W] tensor, producing [1, C, H, W]. |
| `CreateNewInstance` | Creates a new SegNeXt instance with the same configuration but freshly initialized weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads SegNeXt-specific configuration values from a binary stream during model loading. |
| `Dispose(Boolean)` | Releases managed resources held by this SegNeXt instance. |
| `Forward(Tensor<>)` | Executes the full forward pass through the SegNeXt MSCAN encoder and Hamburger decoder. |
| `GetModelConfig(SegNeXtModelSize)` | Returns the architecture configuration for a given SegNeXt model size. |
| `GetModelMetadata` | Collects metadata describing this SegNeXt model's configuration and state. |
| `GetOptions` | Gets the configuration options for this SegNeXt model. |
| `InitializeLayers` | Initializes the MSCAN encoder and Hamburger decoder layers for the SegNeXt model. |
| `PredictCore(Tensor<>)` | Runs a forward pass through the SegNeXt model to produce a per-pixel segmentation map. |
| `PredictOnnx(Tensor<>)` | Runs inference using the ONNX runtime session. |
| `RemoveBatchDimension(Tensor<>)` | Removes the batch dimension from a [1, C, H, W] tensor, producing [C, H, W]. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes all SegNeXt-specific configuration values to a binary stream for persistence. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step: forward pass, loss computation, backward pass, and parameter update. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters across the model from a flat parameter vector. |

