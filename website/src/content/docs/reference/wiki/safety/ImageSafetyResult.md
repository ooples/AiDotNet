---
title: "ImageSafetyResult"
description: "Detailed result from image safety classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Detailed result from image safety classification.

## For Beginners

ImageSafetyResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoryScores` | Per-category safety scores (0.0 = safe, 1.0 = maximum risk). |
| `HighestRiskCategory` | The highest-risk category detected. |
| `HighestRiskScore` | The highest risk score across all categories. |
| `IsSafe` | Whether the image is safe overall. |

