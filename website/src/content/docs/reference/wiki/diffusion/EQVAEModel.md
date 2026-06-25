---
title: "EQVAEModel<T>"
description: "Equivariance-preserving VAE (EQ-VAE) with improved latent regularity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Equivariance-preserving VAE (EQ-VAE) with improved latent regularity.

## For Beginners

If you rotate or flip an image, the latent representation should
rotate or flip in the same way. Standard VAEs don't guarantee this, leading to
inconsistent latent spaces. EQ-VAE adds this guarantee, producing better latents
for diffusion models to work with — like having a well-organized workspace.

## How It Works

EQ-VAE enforces equivariance constraints on the encoder/decoder pair, ensuring that
geometric transformations in pixel space correspond to the same transformations in
latent space. This produces smoother, more regular latent distributions that improve
downstream diffusion model training and generation quality.

Reference: Xu et al., "EQ-VAE: Equivariance Regularized Latent Space for Improved
Generative Image Modeling", 2025

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EQVAEModel(Int32,Int32,Int32,Double,ILossFunction<>,Nullable<Int32>)` | Initializes a new EQ-VAE model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsSlicing` |  |
| `SupportsTiling` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `ComputeEquivarianceLoss(Tensor<>,Tensor<>)` | Computes the equivariance loss between original and transformed encode-decode paths. |
| `Decode(Tensor<>)` |  |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeWithDistribution(Tensor<>)` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

