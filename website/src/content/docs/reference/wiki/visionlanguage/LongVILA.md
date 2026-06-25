---
title: "LongVILA<T>"
description: "LongVILA: long-context visual language model for 1hr+ videos."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

LongVILA: long-context visual language model for 1hr+ videos.

## For Beginners

LongVILA is a video-language model from NVIDIA for understanding
long videos of 1 hour or more. Default values follow the original paper settings.

## How It Works

LongVILA (NVIDIA, 2024) scales long-context visual language models for processing videos
exceeding one hour in duration. It uses multi-modal sequence parallelism to distribute
long video frame sequences across multiple GPUs and extends the context window to handle
thousands of frames for long-form video understanding and temporal reasoning.

**References:**

- Paper: "LongVILA: Scaling Long-Context Visual Language Models for Long Videos" (NVIDIA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using LongVILA's MM-SP single-chunk processing. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using LongVILA's chunked multi-modal sequence processing for long videos (1hr+). |

