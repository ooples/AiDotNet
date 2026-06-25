---
title: "HallucinationResult"
description: "Detailed result from hallucination detection with per-claim verdicts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detailed result from hallucination detection with per-claim verdicts.

## For Beginners

HallucinationResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `Claims` | Per-claim verdicts with explanations. |
| `IsHallucinated` | Whether the content is likely hallucinated. |
| `OverallScore` | Overall hallucination score (0.0 = grounded, 1.0 = fabricated). |

