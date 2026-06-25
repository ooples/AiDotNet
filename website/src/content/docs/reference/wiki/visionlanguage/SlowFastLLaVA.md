---
title: "SlowFastLLaVA<T>"
description: "SlowFast-LLaVA: token-efficient slow/fast pathways for long video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

SlowFast-LLaVA: token-efficient slow/fast pathways for long video.

## For Beginners

SlowFast-LLaVA is a training-free video-language model using
dual pathways for efficient long video understanding. Default values follow the original
paper settings.

## How It Works

SlowFast-LLaVA (Meta, 2025) is a training-free video understanding baseline that uses
dual slow/fast pathways for token-efficient long video processing. The slow pathway processes
fewer frames at high spatial resolution, while the fast pathway samples many frames at low
spatial resolution, combining detailed spatial and broad temporal information efficiently.

**References:**

- Paper: "SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models" (Meta, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using SlowFast-LLaVA's slow pathway at full resolution. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using SlowFast-LLaVA's dual-pathway architecture. |

