---
title: "PointCloudSegmentationOutput<T>"
description: "Output for 3D point cloud segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Output for 3D point cloud segmentation.

## For Beginners

Point cloud segmentation labels each 3D point (from LiDAR, depth
cameras, etc.) with a class. This output contains per-point labels, confidences, and
optional per-point features for downstream tasks like scene understanding.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Per-class names (if available). |
| `ClassPointCounts` | Per-class point counts. |
| `Confidences` | Per-point confidence scores [N]. |
| `Features` | Per-point learned feature vectors [N, featureDim] from the model's encoder. |
| `InferenceTime` | Inference time. |
| `InstanceBoundingBoxes` | Bounding boxes for each detected instance [numInstances, 6] as (x1, y1, z1, x2, y2, z2). |
| `InstanceIds` | Per-point instance IDs [N] for instance segmentation (if supported). |
| `Labels` | Per-point class labels [N]. |
| `Logits` | Per-point class logits [N, numClasses]. |
| `NumClasses` | Number of classes. |
| `NumPoints` | Number of points in the input cloud. |

