---
title: "AudioLoadResult<T>"
description: "Result of loading an audio file, including metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Helpers`

Result of loading an audio file, including metadata.

## Properties

| Property | Summary |
|:-----|:--------|
| `Audio` | Audio samples as tensor [1, channels, samples]. |
| `BitsPerSample` | Bits per sample (8, 16, 24, or 32). |
| `Channels` | Number of channels (1 = mono, 2 = stereo). |
| `DurationSeconds` | Duration in seconds. |
| `SampleRate` | Sample rate in Hz. |

