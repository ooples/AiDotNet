---
title: "RVM<T>"
description: "RVM: Robust Video Matting for real-time human segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Matting`

RVM: Robust Video Matting for real-time human segmentation.

## For Beginners

RVM extracts people from video backgrounds in real-time.
Unlike simple background removal that creates hard edges, matting produces
a soft alpha matte that preserves hair details and semi-transparent regions.

Key capabilities:

- Real-time video matting without green screen
- High-quality alpha matte output
- Temporal consistency across frames
- Works with any background

Outputs:

- Alpha matte: Transparency at each pixel (0=background, 1=foreground)
- Foreground: RGB colors of the person with pre-multiplied alpha

Example usage:

## How It Works

**Technical Details:**

- MobileNetV3 backbone for efficiency
- Recurrent architecture for temporal consistency
- Deep guided filter for detail refinement
- Multi-resolution processing

**Reference:** "Robust High-Resolution Video Matting with Temporal Guidance"
https://arxiv.org/abs/2108.11515

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RVM` | Initializes a new instance with default architecture settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompositeWithBackground(Tensor<>,Tensor<>,Tensor<>)` | Composites the foreground onto a new background. |
| `GetAlpha(Tensor<>)` | Extracts just the alpha matte. |
| `GetOptions` |  |
| `GreenScreenExtract(Tensor<>)` | Creates a green screen effect (extracts foreground). |
| `MatteSingleFrame(Tensor<>)` | Mattes a single frame, maintaining temporal consistency with previous frames. |
| `MatteVideo(List<Tensor<>>)` | Processes a video to extract alpha mattes and foregrounds. |
| `ResetState` | Resets the recurrent state for a new video. |

