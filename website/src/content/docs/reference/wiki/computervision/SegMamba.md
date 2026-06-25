---
title: "SegMamba<T>"
description: "SegMamba: long-range sequential modeling Mamba for 3D medical image segmentation (Xing et al., 2024, arXiv:2401.13560)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

SegMamba: long-range sequential modeling Mamba for 3D medical image segmentation
(Xing et al., 2024, arXiv:2401.13560).

## For Beginners

This model labels every voxel of a 3-D medical scan (e.g. a CT volume) with
the organ/structure it belongs to. It "reads" the whole volume as a long sequence in several
directions so it can relate far-apart regions cheaply.

## How It Works

**Architecture (paper-faithful).** SegMamba is a 3D U-Net whose encoder replaces the usual
self-attention / convolution stack with a Mamba state-space backbone:

The Mamba scan gives linear complexity in the number of voxels, which is what makes whole-volume
3D segmentation tractable where attention would be quadratic.

**Reference:** Xing et al., "SegMamba: Long-range Sequential Modeling Mamba For 3D
Medical Image Segmentation", arXiv:2401.13560, 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegMamba(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,SegMambaOptions)` | Initializes SegMamba in native (trainable) mode. |
| `SegMamba(NeuralNetworkArchitecture<>,String,Int32,SegMambaOptions)` | Initializes SegMamba in ONNX (inference-only) mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyConvBlock(Conv3DLayer<>,InstanceNormalizationLayer<>,Tensor<>)` | Conv → InstanceNorm → ReLU block (decoder). |
| `ApplyGsc(SegMamba<>.GscModule,Tensor<>)` | Gated Spatial Convolution (paper §3.3): two stacked 3×3×3 conv-norm-ReLU branches summed with a 1×1×1 conv-norm-ReLU branch, plus a residual connection. |
| `ApplyTriOrientatedMamba(SegMamba<>.TomModule,Tensor<>)` | Tri-orientated Mamba (ToM, paper §3.2): flatten the 3-D feature volume into a token sequence and scan it with a Mamba SSM in three orientations — forward, reverse, and inter-slice — summing the three results. |
| `ApplyTsMamba(SegMamba<>.TomModule,Tensor<>)` | One TSMamba block: residual + Tri-orientated Mamba over the normalized volume. |
| `CropToSpatial(Tensor<>,Tensor<>)` | Center-crops the spatial dims (D, H, W = axes 2..4) of `t` down to the reference tensor's spatial dims when larger. |
| `ExtractLayerReferences` | Re-derives the typed sub-layer references from the canonical `Layers` list after deserialization rebuilds it. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation logits. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |

