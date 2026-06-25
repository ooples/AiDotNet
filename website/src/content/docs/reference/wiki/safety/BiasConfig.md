---
title: "BiasConfig"
description: "Configuration for bias and fairness detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Fairness`

Configuration for bias and fairness detection modules.

## For Beginners

Use this to configure which types of bias to check for
and which demographic groups to analyze.

## Properties

| Property | Summary |
|:-----|:--------|
| `DisparityThreshold` | Disparity threshold for flagging bias (0.0-1.0). |
| `IntersectionalAnalysis` | Whether to check for intersectional bias. |
| `ProtectedAttributes` | Protected attributes to analyze (e.g., "gender", "race", "age"). |
| `StereotypeDetection` | Whether to detect stereotypical associations. |

