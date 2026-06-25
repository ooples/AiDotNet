---
title: "LISA<T>"
description: "LISA: Reasoning Segmentation via Large Language Model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Referring`

LISA: Reasoning Segmentation via Large Language Model.

## For Beginners

Reasoning-based segmentation from complex text queries. Implicit referring segmentation.

Common use cases:

- Reasoning-based segmentation from complex text queries
- Implicit referring segmentation
- Conversational image segmentation
- World-knowledge-powered segmentation

## How It Works

**Technical Details:**

- Large Language Model (LLaVA) + SAM mask decoder
- Embedding-as-mask paradigm: LLM output tokens control SAM
- Handles complex reasoning queries (not just simple references)
- End-to-end trainable with LoRA fine-tuning

**Reference:** Lai et al., "LISA: Reasoning Segmentation via Large Language Model", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LISA(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,LISAOptions)` | Initializes LISA in native (trainable) mode. |
| `LISA(NeuralNetworkArchitecture<>,String,Int32,LISAOptions)` | Initializes LISA in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this LISA instance supports training. |

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

