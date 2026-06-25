---
title: "ToxicityDetectorConfig"
description: "Configuration for toxicity detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Text`

Configuration for toxicity detection modules.

## For Beginners

Use this to configure how strict the toxicity detector should be.
Higher thresholds mean only very toxic content is flagged; lower thresholds catch
more subtle toxicity but may have more false positives.

## Properties

| Property | Summary |
|:-----|:--------|
| `Categories` | Categories to detect. |
| `IncludeSpanScores` | Whether to include per-span toxicity scores. |
| `Languages` | Languages to support. |
| `Threshold` | Toxicity score threshold (0.0-1.0). |

