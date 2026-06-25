---
title: "SegmentationEvaluation"
description: "Comprehensive evaluation metrics for segmentation models covering all standard benchmarks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Comprehensive evaluation metrics for segmentation models covering all standard benchmarks.

## For Beginners

These metrics measure how good a segmentation model is.
Different metrics matter for different tasks:

- mIoU: Standard for semantic segmentation (higher = better, max 100%)
- AP: Standard for instance segmentation (higher = better, max 100%)
- PQ: Standard for panoptic segmentation (higher = better, max 100%)
- Dice: Standard for medical segmentation (higher = better, max 1.0)
- J&F: Standard for video object segmentation (higher = better, max 100%)

## Properties

| Property | Summary |
|:-----|:--------|
| `AP` | Average Precision averaged over IoU thresholds 0.50:0.05:0.95 (COCO primary metric). |
| `AP50` | Average Precision at IoU=0.50 (COCO-style AP50). |
| `AP75` | Average Precision at IoU=0.75 (stricter). |
| `APLarge` | AP for large objects (area greater than 96^2 pixels). |
| `APMedium` | AP for medium objects (32^2 to 96^2 pixels). |
| `APSmall` | AP for small objects (area less than 32^2 pixels). |
| `AR1` | Average Recall at maximum 1 detection per image. |
| `AR10` | Average Recall at maximum 10 detections per image. |
| `AR100` | Average Recall at maximum 100 detections per image. |
| `BoundaryF1` | Boundary F1 score (precision/recall of boundary pixels). |
| `BoundaryIoU` | Boundary IoU at specified pixel tolerance. |
| `DatasetName` | Name of the evaluation dataset (e.g., "COCO", "ADE20K", "Cityscapes", "DAVIS"). |
| `FPS` | Frames per second. |
| `FScore` | F score (boundary F-measure for VOS). |
| `FrequencyWeightedIoU` | Frequency-weighted IoU (classes weighted by pixel frequency). |
| `GFLOPs` | Floating point operations in GFLOPs. |
| `InferenceTime` | Inference time per image/frame. |
| `JAndFMean` | J&F mean (average of J and F scores). |
| `JScore` | J score (Jaccard/IoU for VOS). |
| `MeanASSD` | Average Symmetric Surface Distance in mm. |
| `MeanDice` | Mean Dice score across classes. |
| `MeanHD95` | Mean Hausdorff Distance (95th percentile) in mm. |
| `MeanIoU` | Mean Intersection over Union across all classes. |
| `MeanNSD` | Normalized Surface Distance at specified tolerance. |
| `MeanPixelAccuracy` | Mean pixel accuracy across classes. |
| `MeanVolumeSimilarity` | Volume Similarity (1 = perfect, 0 = no overlap). |
| `NumClasses` | Number of classes in the evaluation. |
| `NumSamples` | Number of images/frames evaluated. |
| `PQStuff` | PQ for "stuff" (amorphous regions) only. |
| `PQThings` | PQ for "things" (countable objects) only. |
| `PanopticQuality` | Panoptic Quality. |
| `ParameterCountM` | Number of model parameters in millions. |
| `PeakGpuMemoryMB` | Peak GPU memory usage in MB. |
| `PerClassAP` | Per-class AP scores. |
| `PerClassDice` | Per-class Dice scores. |
| `PerClassHD95` | Per-class Hausdorff Distance (95th percentile). |
| `PerClassIoU` | Per-class IoU scores. |
| `PerClassPQ` | Per-class PQ scores. |
| `PerObjectJAndF` | Per-object J&F scores in video. |
| `PixelAccuracy` | Pixel accuracy: fraction of correctly classified pixels. |
| `RecognitionQuality` | Recognition Quality (F1 of matched segments, measures detection accuracy). |
| `SegmentationQuality` | Segmentation Quality (average IoU of matched segments). |
| `TemporalStability` | Temporal stability score (how consistent masks are across frames). |
| `TrimapAccuracy` | Trimap accuracy (accuracy in the uncertain boundary region). |

