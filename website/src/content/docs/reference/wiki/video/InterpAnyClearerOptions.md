---
title: "InterpAnyClearerOptions"
description: "Configuration options for InterpAnyClearer plug-in module for clearer anytime interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for InterpAnyClearer plug-in module for clearer anytime interpolation.

## For Beginners

When objects in a video move at different speeds, standard interpolation
can get confused and produce blurry results. InterpAnyClearer adds a small "speed detector"
that tells the main model how fast each part of the image is moving, so it can produce
sharp results even when some objects move fast and others are still.

## How It Works

InterpAnyClearer (Zheng et al., ECCV 2024 Oral) resolves velocity ambiguity in VFI:

- Velocity-ambiguity analysis: identifies that standard VFI models produce blurry results

when motion speed varies within a scene, because a single flow vector per pixel cannot
represent multiple plausible velocities simultaneously

- Plug-in velocity predictor: a lightweight auxiliary network that predicts per-pixel velocity

magnitude from the input frame pair, conditioning the base VFI model to select the correct
motion hypothesis for each region

- Multi-velocity training: during training, the model sees multiple velocity annotations per

pixel (from different temporal distances), learning to disambiguate fast vs slow motion

- Base-model agnostic: designed as a plug-in that wraps any existing VFI model (RIFE, IFRNet,

AMT, EMA-VFI, etc.) without modifying its architecture, only adding velocity conditioning

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InterpAnyClearerOptions` | Initializes a new instance with default values. |
| `InterpAnyClearerOptions(InterpAnyClearerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels for multi-scale velocity estimation. |
| `NumVelocityBins` | Gets or sets the number of velocity bins for discretized speed estimation. |
| `NumVelocityBlocks` | Gets or sets the number of velocity predictor blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `UseVelocityGuidedWarping` | Gets or sets whether to use velocity-guided warping. |
| `Variant` | Gets or sets the model variant. |

