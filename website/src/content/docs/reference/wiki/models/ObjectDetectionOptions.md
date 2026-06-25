---
title: "ObjectDetectionOptions<T>"
description: "Configuration options for object detection models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for object detection models.

## For Beginners

Object detection finds and locates objects in images.
This class configures how the detection model works, including:

- Which architecture to use (YOLO, DETR, Faster R-CNN, etc.)
- Model size (smaller = faster, larger = more accurate)
- Detection thresholds (how confident the model must be)

## Properties

| Property | Summary |
|:-----|:--------|
| `Architecture` | The detection architecture to use. |
| `Backbone` | The backbone network type for feature extraction. |
| `ClassNames` | Class names for the detection classes. |
| `ConfidenceThreshold` | Minimum confidence score for a detection to be kept. |
| `FreezeBackbone` | Whether to freeze the backbone during fine-tuning. |
| `InputSize` | Input image size [height, width] the model expects. |
| `MaxDetections` | Maximum number of detections to return per image. |
| `Neck` | The neck architecture for multi-scale feature fusion. |
| `NmsThreshold` | IoU threshold for Non-Maximum Suppression (NMS). |
| `NumClasses` | Number of object classes to detect. |
| `RandomSeed` | Random seed for reproducibility. |
| `Size` | The model size variant. |
| `UseMultiScale` | Whether to use multi-scale inference for better accuracy. |
| `UsePretrained` | Whether to use pre-trained weights (COCO dataset). |
| `WeightsUrl` | Custom URL to download weights from. |

