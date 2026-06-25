---
title: "MultimodalOutput<T>"
description: "Represents an output from unified multimodal models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents an output from unified multimodal models.

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Confidence score for the output. |
| `InternalData` | Processed tensor data for this output. |
| `Metadata` | Optional metadata about the output. |
| `Modality` | The modality type of this output. |
| `TextContent` | Text content (for text outputs). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAudioSamples` | Gets audio data as waveform samples for audio outputs. |
| `GetImageDimensions` | Gets the dimensions of image output data. |
| `GetImagePixels` | Gets image data as pixel values for image outputs. |
| `GetVideoDimensions` | Gets the dimensions of video output data. |
| `GetVideoFrames` | Gets video frame data for video outputs. |

