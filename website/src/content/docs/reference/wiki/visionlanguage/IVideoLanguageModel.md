---
title: "IVideoLanguageModel<T>"
description: "Interface for video-language models that process video frames for temporal understanding and QA."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for video-language models that process video frames for temporal understanding and QA.

## How It Works

Video-language models extend image-based VLMs to handle temporal sequences of video frames.
Architectures include frame averaging, spatial-temporal convolution, slow/fast pathways,
and long-context approaches for processing hour+ videos.

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets the name of the language model backbone. |
| `MaxFrames` | Gets the maximum number of video frames the model can process. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from a sequence of video frames, optionally conditioned on a text prompt. |

