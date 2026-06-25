---
title: "SAN<T>"
description: "SAN: Side Adapter Network for open-vocabulary semantic segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.OpenVocabulary`

SAN: Side Adapter Network for open-vocabulary semantic segmentation.

## For Beginners

Open-vocabulary semantic segmentation. Segmenting novel categories from text descriptions.

Common use cases:

- Open-vocabulary semantic segmentation
- Segmenting novel categories from text descriptions
- Zero-shot segmentation
- Language-guided scene understanding

## How It Works

**Technical Details:**

- Side adapter attached to frozen CLIP model
- Recognition head (CLIP) + segmentation head (adapter)
- Two-way attention between adapter and CLIP features
- No re-training of CLIP backbone needed

**Reference:** Xu et al., "Side Adapter Network for Open-Vocabulary Semantic Segmentation", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAN(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,SANOptions)` | Initializes SAN in native (trainable) mode. |
| `SAN(NeuralNetworkArchitecture<>,String,Int32,SANOptions)` | Initializes SAN in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SAN instance supports training. |

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

