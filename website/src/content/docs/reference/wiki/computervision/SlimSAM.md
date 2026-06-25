---
title: "SlimSAM<T>"
description: "SlimSAM: Pruned and distilled SAM for efficient segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Efficient`

SlimSAM: Pruned and distilled SAM for efficient segmentation.

## For Beginners

Efficient segment anything. Pruned SAM for resource-constrained deployment.

Common use cases:

- Efficient segment anything
- Pruned SAM for resource-constrained deployment
- Fast interactive segmentation
- Data-efficient SAM compression

## How It Works

**Technical Details:**

- Alternate slimming: prune + distill iteratively
- Uses only 0.1% of SA-1B data for distillation
- Embedding-disturbed pruning for ViT layers
- Maintains SAM quality with fewer parameters

**Reference:** Chen et al., "SlimSAM: 0.1% Data Frees Slim Segment Anything Model", arXiv 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SlimSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,SlimSAMOptions)` | Initializes SlimSAM in native (trainable) mode. |
| `SlimSAM(NeuralNetworkArchitecture<>,String,Int32,SlimSAMOptions)` | Initializes SlimSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SlimSAM instance supports training. |

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

