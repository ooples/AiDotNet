---
title: "NearMissVersion<T>"
description: "NearMiss variant versions."
section: "API Reference"
---

`Enums` · `AiDotNet.Augmentation.Tabular.Undersampling`

NearMiss variant versions.

## For Beginners

Each version uses a different strategy for selecting
which majority samples to keep.

## Fields

| Field | Summary |
|:-----|:--------|
| `NearMiss1` | Keep majority samples close to nearest minority samples. |
| `NearMiss2` | Keep majority samples close to farthest minority samples. |
| `NearMiss3` | Keep k majority samples nearest to each minority sample. |

