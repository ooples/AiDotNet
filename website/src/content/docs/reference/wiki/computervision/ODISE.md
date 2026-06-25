---
title: "ODISE<T>"
description: "ODISE: Open-vocabulary DIffusion-based panoptic SEgmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Panoptic`

ODISE: Open-vocabulary DIffusion-based panoptic SEgmentation.

## For Beginners

Open-vocabulary panoptic segmentation. Segmenting novel categories not seen during training.

Common use cases:

- Open-vocabulary panoptic segmentation
- Segmenting novel categories not seen during training
- Creative content understanding
- Cross-domain scene parsing

## How It Works

**Technical Details:**

- Leverages Stable Diffusion internal representations for segmentation
- Text-image discriminative model (CLIP) for category assignment
- Mask generator trained on diffusion features
- Zero-shot panoptic segmentation capability

**Reference:** Xu et al., "Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ODISE(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,ODISEModelSize,Double,ODISEOptions)` | Initializes ODISE in native (trainable) mode. |
| `ODISE(NeuralNetworkArchitecture<>,String,Int32,ODISEModelSize,ODISEOptions)` | Initializes ODISE in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this ODISE instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `Forward(Tensor<>)` | Forward pass with Stable Diffusion U-Net skip connections (Xu et al. |
| `ForwardForTraining(Tensor<>)` | Training forward MUST use the same skip-connection path as inference so the lazy decoder convolutions resolve their input channel counts against the CONCATENATED feature maps. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `GetOrCreateBaseOptimizer` | Performs one training step. |
| `InitializeLayers` | Initializes the encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce per-pixel class probabilities. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

