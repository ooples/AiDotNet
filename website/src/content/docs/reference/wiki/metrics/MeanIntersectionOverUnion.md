---
title: "MeanIntersectionOverUnion<T>"
description: "Mean Intersection over Union (mIoU) metric for segmentation tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Mean Intersection over Union (mIoU) metric for segmentation tasks.

## How It Works

mIoU is the standard metric for semantic segmentation evaluation.
It computes IoU for each class and averages across all classes.
IoU = TP / (TP + FP + FN) where TP=true positive, FP=false positive, FN=false negative.

**Usage in 3D AI:**

- Point cloud segmentation (S3DIS, ScanNet)
- Mesh semantic segmentation
- Voxel-based 3D segmentation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeanIntersectionOverUnion(Int32,Boolean)` | Initializes a new instance of the mIoU metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes mIoU between predicted and ground truth segmentation masks. |
| `ComputePerClass(Tensor<>,Tensor<>)` | Computes per-class IoU values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_ignoreBackground` | Whether to ignore the background class (class 0) in computation. |
| `_numClasses` | Number of classes for segmentation. |
| `_numOps` | The numeric operations provider for type T. |

