---
title: "MultimodalInput<T>"
description: "Represents an input item for unified multimodal models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents an input item for unified multimodal models.

## Properties

| Property | Summary |
|:-----|:--------|
| `InternalData` | Processed tensor data for this input. |
| `Metadata` | Optional metadata about the input. |
| `Modality` | The modality type of this input. |
| `SequenceIndex` | Temporal ordering for sequential inputs. |
| `TextContent` | Optional text content (for text modality). |

## Methods

| Method | Summary |
|:-----|:--------|
| `FromAudio(Vector<>,Int32,Int32)` | Creates an audio input from waveform samples. |
| `FromImage(Vector<>,Int32,Int32,Int32,Int32)` | Creates an image input from pixel data. |
| `FromText(String,Int32)` | Creates a text input for the multimodal model. |
| `FromVideo(Vector<>,Int32,Int32,Int32,Int32,Double,Int32)` | Creates a video input from frame data. |

