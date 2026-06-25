---
title: "ToxicityResult"
description: "Detailed result from toxicity detection with per-category scores and spans."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detailed result from toxicity detection with per-category scores and spans.

## For Beginners

ToxicityResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoryScores` | Per-category toxicity scores. |
| `IsToxic` | Whether the content exceeds the configured toxicity threshold. |
| `OverallScore` | Overall toxicity score (0.0 = safe, 1.0 = maximally toxic). |
| `ToxicSpans` | Detected toxic spans with their scores. |

