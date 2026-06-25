---
title: "Stl10DataLoaderOptions"
description: "Configuration options for the STL-10 image classification dataset (Coates et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the STL-10 image classification dataset (Coates et al. 2011).

## How It Works

STL-10 is a 10-class, 96×96 RGB image classification benchmark with 500
labeled train + 800 test images per class, plus 100,000 unlabeled images
for self-supervised pretraining. Used widely for SSL/pretraining studies
since the unlabeled split is large compared to the small labeled split.

## Properties

| Property | Summary |
|:-----|:--------|
| `Normalize` | Normalize byte pixel values to [0, 1] when true (default), or keep raw 0..255 when false. |
| `UseUnlabeled` | When true, returns the unlabeled split (100k images) instead of train/test. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

