---
title: "InterpAnyClearer<T>"
description: "InterpAnyClearer: plug-in module for clearer anytime video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

InterpAnyClearer: plug-in module for clearer anytime video frame interpolation.

## For Beginners

When objects in a video move at different speeds, standard interpolation
can get confused and produce blurry results. InterpAnyClearer adds a small "speed detector"
that tells the main model how fast each part of the image is moving, so it can produce
sharp results even when some objects move fast and others are still.

**Usage:**

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

**Reference:** "Clearer Frames, Anytime: Resolving Velocity Ambiguity in VFI"
(Zheng et al., ECCV 2024 Oral)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InterpAnyClearer(NeuralNetworkArchitecture<>,InterpAnyClearerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an InterpAnyClearer model in native training mode. |
| `InterpAnyClearer(NeuralNetworkArchitecture<>,String,InterpAnyClearerOptions)` | Creates an InterpAnyClearer model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

