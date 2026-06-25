---
title: "SegmentationMetrics<T>"
description: "Provides metrics for evaluating segmentation and classification tasks."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.NeuralNetworks.Metrics`

Provides metrics for evaluating segmentation and classification tasks.

## For Beginners

When a model predicts labels for different parts of a 3D shape
(like "leg", "seat", "back" for a chair), these metrics tell us how accurate those
predictions are compared to the ground truth labels.

## How It Works

These metrics are essential for evaluating the quality of segmentation models
on 3D data like point cloud segmentation, mesh segmentation, and voxel classification.

## Methods

| Method | Summary |
|:-----|:--------|
| `Accuracy(Int32[],Int32[],Int32)` | Computes overall accuracy for classification/segmentation. |
| `ComputeConfusionMatrix(Int32[],Int32[],Int32,Int32)` | Computes the confusion matrix internally. |
| `ConfusionMatrix(Int32[],Int32[],Int32,Int32)` | Computes the confusion matrix. |
| `F1Score(Int32[],Int32[],Int32,Int32)` | Computes F1 score for each class. |
| `MeanF1Score(Int32[],Int32[],Int32,Int32)` | Computes the mean F1 score across all classes. |
| `MeanIoU(Int32[],Int32[],Int32,Int32)` | Computes Mean Intersection over Union (mIoU) for segmentation. |
| `PerClassAccuracy(Int32[],Int32[],Int32,Int32)` | Computes per-class accuracy. |
| `PerClassIoU(Int32[],Int32[],Int32,Int32)` | Computes per-class IoU for segmentation. |
| `Precision(Int32[],Int32[],Int32,Int32)` | Computes precision for each class. |
| `Recall(Int32[],Int32[],Int32,Int32)` | Computes recall for each class. |

