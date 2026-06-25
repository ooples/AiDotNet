---
title: "SamplesSelectedEventArgs<TInput>"
description: "Event arguments for when samples are selected for labeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Interfaces`

Event arguments for when samples are selected for labeling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SamplesSelectedEventArgs(Int32[],[],Double[])` | Initializes a new instance of the SamplesSelectedEventArgs class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InformativenessScores` | Gets the informativeness scores of selected samples. |
| `SelectedIndices` | Gets the indices of selected samples in the unlabeled pool. |
| `SelectedSamples` | Gets the selected samples. |

