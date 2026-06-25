---
title: "QueryMeldNet<T>"
description: "QueryMeldNet (MQ-Former): Dynamic Query Melding for Multi-Dataset Segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

QueryMeldNet (MQ-Former): Dynamic Query Melding for Multi-Dataset Segmentation.

## For Beginners

QueryMeldNet scales mask-based segmentation across multiple diverse datasets
by dynamically melding (fusing) instance queries and stuff queries through cross-attention. This
allows the model to generalize well across different segmentation benchmarks without dataset-specific
fine-tuning.

Common use cases:

- Multi-dataset panoptic segmentation
- Cross-domain segmentation transfer
- Production systems trained on diverse data sources
- Research in universal segmentation scaling

## How It Works

**Technical Details:**

- Dynamic query melding: instance and stuff queries interact via cross-attention layers
- Multi-dataset training with unified query representations
- Backbone: ResNet-50 or Swin-L transformer
- Built on Mask2Former architecture with query interaction extensions

**Reference:** "QueryMeldNet: Dynamic Query Melding for Multi-Dataset Segmentation", CVPR 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryMeldNet(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,QueryMeldNetModelSize,Double,QueryMeldNetOptions)` | Initializes QueryMeldNet in native (trainable) mode. |
| `QueryMeldNet(NeuralNetworkArchitecture<>,String,Int32,Int32,QueryMeldNetModelSize,QueryMeldNetOptions)` | Initializes QueryMeldNet in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this QueryMeldNet instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new QueryMeldNet instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads QueryMeldNet configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this QueryMeldNet model's configuration. |
| `GetOptions` | Gets the configuration options for this QueryMeldNet model. |
| `InitializeLayers` | Initializes the encoder and decoder layers for QueryMeldNet. |
| `PredictCore(Tensor<>)` | Runs a forward pass through QueryMeldNet for multi-dataset segmentation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes QueryMeldNet configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

