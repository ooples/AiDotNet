---
title: "DiffCutSegmentation<T>"
description: "DiffCut: Diffusion-based zero-shot segmentation via graph cuts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Diffusion`

DiffCut: Diffusion-based zero-shot segmentation via graph cuts.

## For Beginners

Zero-shot segmentation from diffusion features. Unsupervised image segmentation.

Common use cases:

- Zero-shot segmentation from diffusion features
- Unsupervised image segmentation
- Open-world object discovery
- Training-free segmentation

## How It Works

**Technical Details:**

- Extracts features from pre-trained Stable Diffusion U-Net
- Recursive Normalized Cut on diffusion feature affinity graph
- No training or fine-tuning required
- Uses diffusion model internal features as dense visual descriptors

**Reference:** Couairon et al., "DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffCutSegmentation(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,DiffCutSegmentationOptions)` | Initializes DiffCutSegmentation in native (trainable) mode. |
| `DiffCutSegmentation(NeuralNetworkArchitecture<>,String,Int32,DiffCutSegmentationOptions)` | Initializes DiffCutSegmentation in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this DiffCutSegmentation instance supports training. |

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

