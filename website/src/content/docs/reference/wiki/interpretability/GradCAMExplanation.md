---
title: "GradCAMExplanation<T>"
description: "Represents the result of a Grad-CAM analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a Grad-CAM analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassScores` | Gets or sets the class scores from the model. |
| `Heatmap` | Gets or sets the upsampled heatmap (same size as input). |
| `InputShape` | Gets or sets the input shape. |
| `IsGradCAMPlusPlus` | Gets or sets whether Grad-CAM++ was used. |
| `OriginalHeatmap` | Gets or sets the original (low-resolution) heatmap from feature maps. |
| `TargetClass` | Gets or sets the target class that was explained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetImportanceAt(Double,Double)` | Gets the heatmap value at a specific location (normalized coordinates). |
| `GetImportantRegions(Double)` | Gets regions with importance above a threshold. |
| `ToString` | Returns a human-readable summary. |

