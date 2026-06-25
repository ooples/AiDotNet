---
title: "VflAlignmentSummary"
description: "Contains summary statistics from the entity alignment phase of VFL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Contains summary statistics from the entity alignment phase of VFL.

## For Beginners

Before VFL training starts, the parties must find which entities
they share. This summary tells you how many entities are shared, how much overlap there is,
and whether there's enough data for meaningful joint training.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignedEntityCount` | Gets or sets the total number of aligned entities across all parties. |
| `AlignmentResult` | Gets or sets the underlying PSI result from the alignment protocol. |
| `AlignmentTime` | Gets or sets the time taken for the alignment phase. |
| `MeetsMinimumOverlap` | Gets or sets whether the alignment meets the minimum overlap threshold. |
| `PartyEntityCounts` | Gets or sets the per-party entity counts before alignment. |
| `PartyOverlapRatios` | Gets or sets the per-party overlap ratios (fraction of party's entities that are aligned). |

