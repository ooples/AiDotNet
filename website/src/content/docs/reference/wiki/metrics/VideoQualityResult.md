---
title: "VideoQualityResult<T>"
description: "Results from comprehensive video quality evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Results from comprehensive video quality evaluation.

## Properties

| Property | Summary |
|:-----|:--------|
| `FlickerScore` | Flicker score (0-1, lower is better). |
| `MaxPSNR` | Maximum per-frame PSNR (dB). |
| `MeanPSNR` | Mean PSNR across all frames (dB). |
| `MeanSSIM` | Mean SSIM across all frames (0-1). |
| `MinPSNR` | Minimum per-frame PSNR (dB). |
| `OverallScore` | Overall composite quality score (0-1). |
| `PerFramePSNR` | PSNR values for each frame. |
| `TemporalConsistency` | Temporal consistency score (0-1, higher is better). |

