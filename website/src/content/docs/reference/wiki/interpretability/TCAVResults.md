---
title: "TCAVResults<T>"
description: "Represents TCAV results for multiple concepts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents TCAV results for multiple concepts.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TCAVResults(List<TCAVResult<>>,Int32)` | Initializes new TCAV results. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NegativeInfluence` | Gets results for concepts with negative influence. |
| `PositiveInfluence` | Gets results for concepts with positive influence. |
| `Results` | Gets all concept results, sorted by significance. |
| `SignificantResults` | Gets results for only significant concepts. |
| `TargetClass` | Gets the target class that was analyzed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary. |

