---
title: "HelixOptions"
description: "Configuration options for Helix."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Robotics`

Configuration options for Helix.

## For Beginners

These options configure the Helix model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HelixOptions(HelixOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumJoints` | Number of joint DOFs the upper-body controller exposes. |
| `System1HiddenDim` | Hidden dimension of the System 1 fast visuomotor transformer. |
| `System1NumHeads` | Number of attention heads per System 1 transformer block. |
| `System1NumLayers` | Number of transformer blocks in System 1. |
| `System1ToSystem2Ratio` | How many S1 ticks one S2 invocation remains valid before the runner re-invokes S2. |
| `System2LatentDim` | Dimensionality of the latent vector System 2 emits to condition System 1. |
| `WeightOffloadOptions` | Optional weight-offload / streaming configuration. |

