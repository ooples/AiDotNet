---
title: "SegmentationSample<T>"
description: "Represents a segmentation dataset sample with image, masks, and metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Represents a segmentation dataset sample with image, masks, and metadata.

## For Beginners

A segmentation dataset contains pairs of images and their
ground-truth masks. This class represents a single sample that can be used for
training, evaluation, or visualization. It supports all segmentation task types
including semantic labels, instance masks, and panoptic annotations.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Class name mapping. |
| `DatasetName` | Dataset name this sample belongs to. |
| `Height` | Image height in pixels. |
| `Image` | Input image tensor [C, H, W]. |
| `InstanceBoxes` | Instance bounding boxes [numInstances, 4] as (x1, y1, x2, y2). |
| `InstanceClassIds` | Instance class IDs [numInstances]. |
| `InstanceMasks` | Ground-truth instance masks [numInstances, H, W] (for instance/panoptic). |
| `NumClasses` | Number of classes in the dataset. |
| `PointCloud` | 3D point cloud [N, featureDim] (for point cloud segmentation). |
| `SampleId` | Sample identifier (e.g., file name or dataset ID). |
| `SemanticLabels` | Ground-truth semantic label map [H, W] (for semantic/panoptic). |
| `TextAnnotations` | Text annotations associated with this sample (for referring segmentation). |
| `VideoFrames` | Video frames [numFrames, C, H, W] (for video segmentation). |
| `VideoMasks` | Per-frame masks [numFrames, numObjects, H, W] (for video segmentation). |
| `Volume` | 3D volume [C, D, H, W] (for medical volumetric segmentation). |
| `Width` | Image width in pixels. |

