---
title: "SSLAugmentationContext<T>"
description: "Context for SSL augmentation operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Context for SSL augmentation operations.

## For Beginners

This provides information and state for creating augmented views.
SSL methods typically create multiple views of each input for contrastive/self-supervised learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsFirstView` | Gets or sets whether this is the first view (used in multi-view methods). |
| `PrecomputedView` | Gets or sets optional pre-computed augmentations. |
| `Seed` | Gets or sets the random seed for reproducible augmentations. |
| `StrengthMultiplier` | Gets or sets the augmentation strength multiplier. |
| `TotalViews` | Gets or sets the total number of views being generated. |
| `ViewIndex` | Gets or sets the view index for methods that use more than 2 views. |

