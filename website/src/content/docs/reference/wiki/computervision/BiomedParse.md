---
title: "BiomedParse<T>"
description: "BiomedParse: Biomedical image parsing with text prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

BiomedParse: Biomedical image parsing with text prompts.

## For Beginners

Text-prompted biomedical image segmentation. Multi-modality biomedical parsing.

Common use cases:

- Text-prompted biomedical image segmentation
- Multi-modality biomedical parsing
- Detection and recognition in biomedical images
- Joint segmentation-detection-recognition

## How It Works

**Technical Details:**

- Text-prompted segmentation for biomedical images
- Joint segmentation, detection, and recognition in one model
- Trained on 6M+ triples across 9 imaging modalities
- GPT-4 assisted harmonization of biomedical datasets

**Reference:** Zhao et al., "BiomedParse: a biomedical foundation model for image parsing of everything everywhere all at once", Nature Methods 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiomedParse(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,BiomedParseOptions)` | Initializes BiomedParse in native (trainable) mode. |
| `BiomedParse(NeuralNetworkArchitecture<>,String,Int32,BiomedParseOptions)` | Initializes BiomedParse in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this BiomedParse instance supports training. |

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

