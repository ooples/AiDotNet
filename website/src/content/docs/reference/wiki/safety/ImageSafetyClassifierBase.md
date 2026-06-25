---
title: "ImageSafetyClassifierBase<T>"
description: "Abstract base class for image safety classifiers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Image`

Abstract base class for image safety classifiers.

## For Beginners

This base class provides common code for all image safety
classifiers. Each classifier type extends this and adds its own way of detecting
harmful content in images.

## How It Works

Provides shared infrastructure for image safety classifiers including threshold
configuration. Concrete implementations provide the actual classification
algorithm (CLIP, ViT, scene graph, ensemble).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageSafetyClassifierBase(Double)` | Initializes the image safety classifier base with a threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetCategoryScores(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `Threshold` | The safety threshold above which content is flagged. |

