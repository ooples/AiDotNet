---
title: "SED<T>"
description: "SED: A Simple Encoder-Decoder for open-vocabulary semantic segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.OpenVocabulary`

SED: A Simple Encoder-Decoder for open-vocabulary semantic segmentation.

## For Beginners

Simple and efficient open-vocabulary segmentation. Category-adaptive prompting.

Common use cases:

- Simple and efficient open-vocabulary segmentation
- Category-adaptive prompting
- Cross-dataset generalization
- Balanced accuracy-efficiency open-vocab segmentation

## How It Works

**Technical Details:**

- Simple encoder-decoder with hierarchical CLIP features
- Category-adaptive prompt generation
- Lightweight decoder head for open-vocabulary prediction
- Competitive accuracy with minimal architectural overhead

**Reference:** Xie et al., "SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SED(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,SEDOptions)` | Initializes SED in native (trainable) mode. |
| `SED(NeuralNetworkArchitecture<>,String,Int32,SEDOptions)` | Initializes SED in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SED instance supports training. |

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

