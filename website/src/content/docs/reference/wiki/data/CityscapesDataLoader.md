---
title: "CityscapesDataLoader<T>"
description: "Loads the Cityscapes semantic-segmentation dataset (Cordts et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Cityscapes semantic-segmentation dataset (Cordts et al. 2016).

## How It Works

Expects the canonical extracted layout (after manually downloading
both archives from cityscapes-dataset.com — sign-up required, no
auto-download):

**Output shape:** features Tensor[N, H, W, 3] (resized RGB image,
normalized to [0, 1]); labels Tensor[N, H, W] (integer class IDs per
pixel — the standard semantic-seg label format used by PyTorch /
TF / SegFormer / DeepLab loss heads). With `MapToTrainIds`
(default) source IDs 0..33 are remapped to the 19 evaluation classes;
"ignore" pixels get sentinel value 255.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `IdToTrainId` | Cityscapes ID-to-trainID lookup (Cordts et al. |
| `IgnoreLabel` | Sentinel ID for "ignore" pixels (background, out-of-eval classes). |
| `NumEvalClasses` | Number of evaluation classes when MapToTrainIds is true. |

