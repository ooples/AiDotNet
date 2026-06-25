---
title: "AugmentationPolicy<T>"
description: "Defines a serializable augmentation policy with named transforms and parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Defines a serializable augmentation policy with named transforms and parameters.

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IAugmentation<,ImageTensor<>>,Double)` | Adds a transform to the policy. |
| `Apply(ImageTensor<>,AugmentationContext<>)` | Applies the policy to an image. |
| `GetParameters` | Gets all parameters as a serializable dictionary. |

