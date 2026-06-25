---
title: "SegmentationOutput<T>"
description: "Unified output type for all segmentation tasks, combining semantic, instance, and panoptic results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Unified output type for all segmentation tasks, combining semantic, instance, and panoptic results.

## For Beginners

This is the universal result type returned by segmentation models.
Depending on the model, different fields will be populated:

- Semantic models fill `ClassMap` and `ClassProbabilities`
- Instance models fill `InstanceMasks` and `InstanceClasses`
- Panoptic models fill both semantic and instance fields plus `PanopticMap`

Use `TaskType` to determine which fields are available.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassMap` | Per-pixel class label map [H, W]. |
| `ClassNames` | Class name labels (if available). |
| `ClassProbabilities` | Per-pixel class probability map [numClasses, H, W]. |
| `ImageHeight` | Input image height. |
| `ImageWidth` | Input image width. |
| `InferenceTime` | Time taken for inference. |
| `InstanceBoxes` | Bounding boxes for each instance [numInstances, 4] as (x1, y1, x2, y2). |
| `InstanceClasses` | Class ID for each detected instance [numInstances]. |
| `InstanceIdMap` | Per-pixel instance ID map [H, W]. |
| `InstanceMasks` | Per-instance binary masks [numInstances, H, W]. |
| `InstanceScores` | Confidence score for each detected instance [numInstances]. |
| `Logits` | Raw logits output [numClasses, H, W] before softmax/argmax. |
| `NumClasses` | Number of classes in the output. |
| `NumInstances` | Number of detected instances, derived from InstanceMasks (preferred) or InstanceClasses. |
| `PanopticMap` | Combined panoptic ID map [H, W] encoded as classId * 1000 + instanceId. |
| `Segments` | Panoptic segment metadata. |
| `TaskType` | The segmentation task that produced this output. |

