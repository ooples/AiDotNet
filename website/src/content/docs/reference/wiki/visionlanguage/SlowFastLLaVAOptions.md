---
title: "SlowFastLLaVAOptions"
description: "Configuration options for SlowFast-LLaVA: token-efficient slow/fast pathways for long video."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.VideoLanguage`

Configuration options for SlowFast-LLaVA: token-efficient slow/fast pathways for long video.

## For Beginners

These options configure the SlowFastLLaVA model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SlowFastLLaVAOptions(SlowFastLLaVAOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FastFrames` | Gets or sets the number of fast pathway frames (low-detail). |
| `SlowFrames` | Gets or sets the number of slow pathway frames (high-detail). |

