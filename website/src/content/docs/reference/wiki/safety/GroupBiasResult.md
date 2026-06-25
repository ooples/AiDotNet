---
title: "GroupBiasResult"
description: "Bias analysis result for a specific demographic group."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Fairness`

Bias analysis result for a specific demographic group.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attribute` | The protected attribute category (e.g., "gender", "race"). |
| `Disparity` | Disparity from the overall mean sentiment. |
| `Group` | The demographic group name. |
| `SentimentScore` | Sentiment score for this group (-1.0 to 1.0). |

