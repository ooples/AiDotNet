---
title: "DiffSeg<T>"
description: "DiffSeg: Unsupervised Semantic Segmentation from Diffusion Model Self-Attention Maps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Semantic`

DiffSeg: Unsupervised Semantic Segmentation from Diffusion Model Self-Attention Maps.

## For Beginners

DiffSeg produces segmentation maps without any training labels by
leveraging the self-attention maps inside a diffusion model. The idea is that diffusion
models learn to "attend" to semantically similar regions when generating images, and
DiffSeg repurposes those attention patterns to group pixels into coherent segments.

Common use cases:

- Unsupervised image segmentation (no labels needed at all)
- Automatic annotation/pre-labeling for new datasets
- Understanding what a diffusion model has learned about image structure
- Research into emergent visual representations in generative models

## How It Works

**Technical Details:**

- Extracts self-attention maps from a pre-trained Stable Diffusion UNet
- Merges attention heads and layers into a single affinity matrix
- Applies iterative attention map merging to produce coherent segments
- Completely training-free: uses frozen diffusion model weights only

**Reference:** Tian et al., "Diffuse, Attend, and Segment: Unsupervised Zero-Shot
Segmentation using Stable Diffusion", arXiv 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffSeg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,DiffSegOptions)` | Initializes DiffSeg in native (trainable) mode. |
| `DiffSeg(NeuralNetworkArchitecture<>,String,Int32,DiffSegOptions)` | Initializes DiffSeg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this DiffSeg instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelMetadata` | Collects model metadata. |
| `InitializeLayers` | Initializes the diffusion attention encoder and segment classification decoder. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce per-pixel segmentation logits. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat vector. |

