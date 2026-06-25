---
title: "ControlNeXtModel<T>"
description: "ControlNeXt model with improved efficiency and generalization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNeXt model with improved efficiency and generalization.

## For Beginners

ControlNeXt is a newer, more efficient version of ControlNet.
It uses a smarter design that requires fewer parameters and less memory while
working just as well (or better) at following control signals.

## How It Works

ControlNeXt improves upon ControlNet by using a more parameter-efficient architecture
with cross-normalization layers instead of a full encoder copy. This reduces memory
usage while improving generalization across different control types.

Reference: Peng et al., "ControlNeXt: Powerful and Efficient Control for Image and Video Generation", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

