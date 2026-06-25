---
title: "LabelMixingEventArgs<T>"
description: "Event arguments for label mixing operations in augmentations like Mixup and CutMix."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Event arguments for label mixing operations in augmentations like Mixup and CutMix.

## For Beginners

In normal classification, an image is 100% one class (like "cat").
With Mixup/CutMix, we blend two images together, so the label becomes something like
"70% cat, 30% dog". This helps the model learn smoother decision boundaries.

## How It Works

When augmentations like Mixup or CutMix are applied, they blend data from two samples
together. The labels must also be blended proportionally. This event allows the training
loop to handle the soft label generation appropriately.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LabelMixingEventArgs(Vector<>,Vector<>,,Int32,Int32,MixingStrategy)` | Creates a new label mixing event. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets additional metadata about the mixing operation. |
| `MixedLabels` | Gets or sets the resulting mixed soft labels. |
| `MixingLambda` | Gets the mixing coefficient (lambda). |
| `OriginalLabels1` | Gets the original hard labels for the first sample. |
| `OriginalLabels2` | Gets the original hard labels for the second (mixed) sample. |
| `SampleIndex1` | Gets the index of the first sample in the batch. |
| `SampleIndex2` | Gets the index of the second sample in the batch. |
| `Strategy` | Gets the mixing strategy used. |

