---
title: "DiffCut<T>"
description: "DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Semantic`

DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut.

## For Beginners

DiffCut is a unique semantic segmentation model that requires no training
labels at all. It extracts features from a diffusion model's internal UNet representations and
applies a graph-based algorithm called Normalized Cut to partition the image into semantically
meaningful regions. This "zero-shot" approach means you can segment images without ever training
on segmentation labels.

Common use cases:

- Zero-shot segmentation when no labeled data is available
- Exploring and annotating new datasets
- Domain adaptation where labeled data doesn't exist
- Research into unsupervised visual understanding

## How It Works

**Technical Details:**

- Extracts intermediate features from a pre-trained Stable Diffusion UNet
- Builds an affinity graph from diffusion feature similarities
- Applies recursive Normalized Cut (NCut) for hierarchical segmentation
- Achieves +7.3 mIoU over prior SOTA on unsupervised segmentation benchmarks
- Training-free: no fine-tuning required

**Reference:** Couairon et al., "DiffCut: Catalyzing Zero-Shot Semantic Segmentation
with Diffusion Features and Recursive Normalized Cut", NeurIPS 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffCut(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,DiffCutOptions)` | Initializes DiffCut in native (trainable) mode. |
| `DiffCut(NeuralNetworkArchitecture<>,String,Int32,DiffCutOptions)` | Initializes DiffCut in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this DiffCut instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelMetadata` | Collects model metadata. |
| `InitializeLayers` | Initializes the diffusion UNet encoder and Normalized Cut decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce per-pixel segmentation logits. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat vector. |

