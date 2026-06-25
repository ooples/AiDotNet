---
title: "E2FGVI<T>"
description: "E2FGVI - End-to-End Framework for Flow-Guided Video Inpainting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Inpainting`

E2FGVI - End-to-End Framework for Flow-Guided Video Inpainting.

## For Beginners

E2FGVI removes unwanted objects from videos and fills in the gaps
with realistic content. It uses optical flow (motion information) to propagate known
content into missing regions across frames.

Use cases:

- Remove watermarks or logos from videos
- Remove unwanted people or objects
- Repair damaged or corrupted video frames
- Video restoration and cleanup

## How It Works

**Technical Details:**

- End-to-end trainable flow-guided inpainting
- Bidirectional flow propagation
- Transformer-based content hallucination
- Temporal consistency enforcement

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `E2FGVI` | Initializes a new instance with default architecture settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputHeight` | Gets the input frame height. |
| `InputWidth` | Gets the input frame width. |
| `SupportsTraining` | Gets whether training is supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateMaskFromDifference(Tensor<>,Tensor<>)` | Creates a mask based on differences between input and expected output. |
| `GetOptions` |  |
| `Inpaint(List<Tensor<>>,List<Tensor<>>)` | Inpaints a video sequence by filling in masked regions. |
| `Inpaint(Tensor<>,Tensor<>)` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` | Predicts the inpainted output for a single masked frame. |
| `PreprocessFrames(Tensor<>)` |  |
| `RemoveObject(List<Tensor<>>,List<Tensor<>>)` | Removes an object from video based on mask sequence. |
| `RepairVideo(List<Tensor<>>,List<Tensor<>>)` | Repairs corrupted regions in video frames. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a masked frame and its ground truth. |

