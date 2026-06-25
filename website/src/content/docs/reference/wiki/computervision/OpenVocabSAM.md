---
title: "OpenVocabSAM<T>"
description: "Open-Vocabulary SAM: SAM with text-based open-vocabulary recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.OpenVocabulary`

Open-Vocabulary SAM: SAM with text-based open-vocabulary recognition.

## For Beginners

Text-prompted SAM segmentation. Open-vocabulary interactive segmentation.

Common use cases:

- Text-prompted SAM segmentation
- Open-vocabulary interactive segmentation
- Large-vocabulary object segmentation
- Combining SAM masks with CLIP recognition

## How It Works

**Technical Details:**

- SAM encoder-decoder with CLIP text alignment
- Supports 20,000+ categories via open vocabulary
- Interactive segmentation with automatic class labels
- Region-level CLIP feature extraction for classification

**Reference:** Yuan et al., "Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively", ECCV 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenVocabSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,OpenVocabSAMOptions)` | Initializes OpenVocabSAM in native (trainable) mode. |
| `OpenVocabSAM(NeuralNetworkArchitecture<>,String,Int32,OpenVocabSAMOptions)` | Initializes OpenVocabSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this OpenVocabSAM instance supports training. |

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

