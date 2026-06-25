---
title: "ISelfPacedScheduler<T>"
description: "Interface for self-paced curriculum schedulers that adapt based on model performance."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Interface for self-paced curriculum schedulers that adapt based on model performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `GrowthRate` | Gets or sets the growth rate for the pace parameter. |
| `PaceParameter` | Gets or sets the pace parameter (lambda in self-paced learning). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSampleWeights(Vector<>)` | Computes sample weights for self-paced learning. |

