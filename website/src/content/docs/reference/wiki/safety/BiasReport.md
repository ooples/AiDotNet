---
title: "BiasReport"
description: "Detailed report from bias and fairness evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Fairness`

Detailed report from bias and fairness evaluation.

## For Beginners

BiasReport provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `BiasDetected` | Whether any bias was detected above the threshold. |
| `GroupResults` | Per-group analysis results. |
| `OverallBiasScore` | Overall bias score (0.0 = fair, 1.0 = severely biased). |
| `StereotypesDetected` | Detected stereotypical associations. |

