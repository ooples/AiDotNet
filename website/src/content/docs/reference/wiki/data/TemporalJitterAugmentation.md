---
title: "TemporalJitterAugmentation<T>"
description: "Applies temporal jitter to video frame sequences by randomly shifting frame indices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms`

Applies temporal jitter to video frame sequences by randomly shifting frame indices.

## How It Works

Temporal jitter is a data augmentation technique for video models that randomly
perturbs the starting position or sampling stride of frames. This forces the model
to be robust to temporal alignment variations and prevents overfitting to specific
frame positions within video clips.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalJitterAugmentation(Int32,Nullable<Int32>)` | Initializes a new temporal jitter augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Applies temporal jitter by circularly shifting frame data. |

