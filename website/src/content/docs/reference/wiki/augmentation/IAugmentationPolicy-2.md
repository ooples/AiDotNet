---
title: "IAugmentationPolicy<T, TData>"
description: "Interface for composable augmentation policies."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Interface for composable augmentation policies.

## How It Works

Policies define collections of augmentations and their application strategies.

## Properties

| Property | Summary |
|:-----|:--------|
| `Augmentations` | Gets the augmentations in this policy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(,AugmentationContext<>)` | Applies the policy to input data. |
| `GetConfiguration` | Gets the parameters of this policy for serialization. |

