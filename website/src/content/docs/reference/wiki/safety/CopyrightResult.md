---
title: "CopyrightResult"
description: "Detailed result from copyright and memorization detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detailed result from copyright and memorization detection.

## For Beginners

CopyrightResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsCopyrightViolation` | Whether the content likely infringes copyright. |
| `Matches` | Detected overlapping segments with known copyrighted content. |
| `MemorizationScore` | Overall memorization score (0.0 = original, 1.0 = memorized). |

