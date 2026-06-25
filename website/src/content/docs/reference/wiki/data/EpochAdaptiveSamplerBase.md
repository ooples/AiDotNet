---
title: "EpochAdaptiveSamplerBase<T>"
description: "Base class for epoch-adaptive samplers that change behavior over training epochs."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Sampling`

Base class for epoch-adaptive samplers that change behavior over training epochs.

## How It Works

EpochAdaptiveSamplerBase is for samplers like curriculum learning and self-paced
learning that adjust their sampling strategy based on the current epoch.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EpochAdaptiveSamplerBase(Int32,Nullable<Int32>)` | Initializes a new instance of the EpochAdaptiveSamplerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Progress` | Gets the current progress through the curriculum (0.0 to 1.0). |

## Methods

| Method | Summary |
|:-----|:--------|
| `OnEpochStart(Int32)` |  |
| `OnEpochStartCore(Int32)` | Called when a new epoch starts. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `TotalEpochs` | The total number of epochs for curriculum progression. |

