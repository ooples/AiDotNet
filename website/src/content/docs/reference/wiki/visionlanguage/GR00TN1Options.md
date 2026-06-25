---
title: "GR00TN1Options"
description: "Configuration options for GR00T N1."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Robotics`

Configuration options for GR00T N1.

## For Beginners

These options configure the GR00TN1 model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GR00TN1Options(GR00TN1Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FlowMatchingSteps` | Number of Euler integration steps the flow-matching action head runs at inference. |
| `NumJoints` | Number of humanoid joints. |
| `System1HiddenDim` | Hidden dimension of the System-1 DiT velocity transformer. |
| `System1NumHeads` | Number of attention heads per System-1 DiT block. |
| `System1NumLayers` | Number of DiT transformer blocks in System 1. |
| `System1ToSystem2Ratio` | How many S1 ticks an S2 latent remains valid in streaming mode. |
| `System2LatentDim` | Dimensionality of the latent vector System 2 (Eagle-2 VLM) emits to condition the flow-matching head. |

