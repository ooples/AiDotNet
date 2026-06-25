---
title: "HarmfulContentResult<T>"
description: "Result of harmful content identification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Result of harmful content identification.

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoryScores` | Gets or sets scores for each harmful content category. |
| `Details` | Gets or sets additional detection details. |
| `DetectedCategories` | Gets or sets all harmful categories detected above threshold. |
| `Findings` | Gets or sets detailed harmful content findings. |
| `HarmScore` | Gets or sets the overall harm score (0-1, higher means more harmful). |
| `HarmfulContentDetected` | Gets or sets whether harmful content was detected. |
| `PrimaryHarmCategory` | Gets or sets the primary harmful category detected. |
| `RecommendedAction` | Gets or sets the recommended action based on the harm level. |

