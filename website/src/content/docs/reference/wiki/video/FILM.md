---
title: "FILM<T>"
description: "FILM (Frame Interpolation for Large Motion) model for high-quality frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

FILM (Frame Interpolation for Large Motion) model for high-quality frame interpolation.

## For Beginners

FILM generates smooth intermediate frames between two input frames,
even when there's significant motion between them. It's particularly good at:

- Large motion scenes (fast camera movements, rapid object motion)
- Creating slow-motion effects from regular video
- Increasing video frame rate (24fps to 60fps)
- Smooth transitions between keyframes

Unlike older methods that struggle with large motions, FILM uses a multi-scale
feature extraction approach that handles both small and large movements gracefully.

## How It Works

**Technical Details:**

- Multi-scale feature pyramid for handling large motions
- Bi-directional flow estimation with occlusion handling
- Feature-based frame synthesis (not just flow warping)
- Scale-agnostic architecture for arbitrary resolution

**Reference:** Reda et al., "FILM: Frame Interpolation for Large Motion"
ECCV 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FILM` | Initializes a new instance of the FILM class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputHeight` | Gets the input frame height. |
| `InputWidth` | Gets the input frame width. |
| `NumScales` | Gets the number of pyramid scales. |
| `SupportsTraining` | Gets whether training is supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateSlowMotion(List<Tensor<>>,Int32)` | Creates slow-motion effect from video frames. |
| `GetOptions` |  |
| `IncreaseFrameRate(List<Tensor<>>,Int32)` | Increases video frame rate by a given factor. |
| `Interpolate(Tensor<>,Tensor<>,Double)` | Interpolates a frame between two input frames. |
| `InterpolateMultiple(Tensor<>,Tensor<>,Int32)` | Generates multiple intermediate frames between two input frames. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |

