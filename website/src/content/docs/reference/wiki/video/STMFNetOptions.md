---
title: "STMFNetOptions"
description: "Configuration options for STMFNet spatio-temporal multi-flow network."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for STMFNet spatio-temporal multi-flow network.

## For Beginners

Instead of guessing one "best" motion for each pixel, STMFNet makes
multiple guesses (flows) and then picks the best one for each part of the image. This works
much better in tricky areas like where objects overlap or where motion is ambiguous.

## How It Works

STMFNet (2022) uses multiple optical flows in spatio-temporal space:

- Multi-flow estimation: estimates multiple (typically 4) optical flow fields between the

input frames, each capturing different motion hypotheses for ambiguous regions like
occlusion boundaries, transparent objects, and repeating textures

- Spatio-temporal feature volume: constructs a 4D (height x width x time x channel) feature

volume from the input frames and all estimated flow fields, capturing the full motion
context in a unified representation

- Flow selection network: a learned network that selects the best flow hypothesis for each

pixel by comparing warped features from each flow field, choosing the one that produces
the most consistent result

- Residual refinement: after flow-based warping, a refinement network corrects remaining

artifacts using the multi-flow feature volume as context

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STMFNetOptions` | Initializes a new instance with default values. |
| `STMFNetOptions(STMFNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFlowHypotheses` | Gets or sets the number of flow hypotheses per pixel. |
| `NumFusionBlocks` | Gets or sets the number of spatio-temporal fusion blocks. |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels for multi-scale flow. |
| `NumRefineBlocks` | Gets or sets the number of refinement blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

