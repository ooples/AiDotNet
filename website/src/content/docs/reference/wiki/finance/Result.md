---
title: "Result"
description: "Result of the BH procedure: per-hypothesis rejection flags and adjusted q-values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Evaluation`

Result of the BH procedure: per-hypothesis rejection flags and adjusted q-values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Result(IReadOnlyList<Boolean>,IReadOnlyList<Double>,Int32)` | Creates a BH result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumRejected` | Number of rejected hypotheses (discoveries). |
| `QValues` | BH-adjusted q-values, aligned to the input p-value order, each in [0, 1]. |
| `Rejected` | Rejected[i] is true when hypothesis i is rejected (a discovery) at the given alpha. |

