---
title: "GLaMM<T>"
description: "GLaMM: Grounding Large Multimodal Model for pixel-level understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Referring`

GLaMM: Grounding Large Multimodal Model for pixel-level understanding.

## For Beginners

Grounded conversation about images. Pixel-level visual understanding with language.

Common use cases:

- Grounded conversation about images
- Pixel-level visual understanding with language
- Region-specific captioning and segmentation
- Multi-turn visual dialogue with grounding

## How It Works

**Technical Details:**

- Grounding LMM with pixel-level output capability
- Generates text with embedded segmentation masks
- Region-level and pixel-level visual features
- Trained on Grounding-anything Dataset (GranD)

**Reference:** Rasheed et al., "GLaMM: Pixel Grounding Large Multimodal Model", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GLaMM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,GLaMMOptions)` | Initializes GLaMM in native (trainable) mode. |
| `GLaMM(NeuralNetworkArchitecture<>,String,Int32,GLaMMOptions)` | Initializes GLaMM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this GLaMM instance supports training. |

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

