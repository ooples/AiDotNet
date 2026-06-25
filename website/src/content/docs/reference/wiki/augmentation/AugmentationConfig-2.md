---
title: "AugmentationConfig<T, TInput>"
description: "Strongly-typed augmentation configuration parameterised by the model's numeric type and input type."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Augmentation`

Strongly-typed augmentation configuration parameterised by the model's
numeric type and input type. Exposes a compile-time-checked
`Augmenter` slot in place of the base class's `object?``CustomAugmenter` property — IntelliSense
guides callers to a valid `IAugmentation`
implementation and the compiler rejects type-mismatch at the call site
instead of failing as a runtime cast deep inside
`AiModelBuilder.BuildSupervisedInternalAsync` (review #1368).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentationConfig` | Default-constructed strongly-typed configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Augmenter` | Strongly-typed augmenter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForAudio` | Audio-modality preset. |
| `ForImages` | Image-modality preset with industry-standard defaults. |
| `ForTabular` | Tabular-modality preset. |
| `ForText` | Text-modality preset. |
| `ForVideo` | Video-modality preset. |

