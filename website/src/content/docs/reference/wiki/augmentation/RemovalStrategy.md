---
title: "RemovalStrategy<T>"
description: "Strategy for removing samples in Tomek links."
section: "API Reference"
---

`Enums` · `AiDotNet.Augmentation.Tabular`

Strategy for removing samples in Tomek links.

## For Beginners

Determines which sample(s) to remove when a Tomek link is found.

## Fields

| Field | Summary |
|:-----|:--------|
| `RemoveBoth` | Remove both samples in the Tomek link. |
| `RemoveMajority` | Only remove the majority class sample (recommended for imbalanced data). |
| `RemoveMinority` | Only remove the minority class sample (rarely used). |

