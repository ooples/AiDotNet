---
title: "Fold"
description: "One purged/embargoed walk-forward fold: the surviving train indices and the test indices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Evaluation`

One purged/embargoed walk-forward fold: the surviving train indices and the test indices.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Fold(IReadOnlyList<Int32>,IReadOnlyList<Int32>)` | Creates a fold from its train and test index lists. |

## Properties

| Property | Summary |
|:-----|:--------|
| `TestIndices` | Test-sample indices for this fold (a contiguous forward block). |
| `TrainIndices` | Training-sample indices remaining after purge and embargo. |

