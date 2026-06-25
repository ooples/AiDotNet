---
title: "MedicalSegmentationOutput<T>"
description: "Output for medical image segmentation with volumetric support and clinical metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Output for medical image segmentation with volumetric support and clinical metadata.

## For Beginners

Medical segmentation output includes organ/structure labels for
each pixel or voxel, along with clinical metrics like Dice scores and volume measurements
that doctors use to assess segmentation quality and make clinical decisions.

## Properties

| Property | Summary |
|:-----|:--------|
| `DiceScores` | Per-class Dice scores (if ground truth was provided during evaluation). |
| `HausdorffDistances` | Per-class Hausdorff distances in mm (if ground truth was provided). |
| `InferenceTime` | Inference time. |
| `Is3D` | Whether this result is from a 3D volumetric segmentation. |
| `Labels` | Per-pixel/voxel class labels. |
| `Modality` | Imaging modality used (CT, MRI_T1, MRI_T2, etc.). |
| `OrientationMatrix` | Image orientation matrix (for DICOM compatibility). |
| `OverallConfidence` | Model confidence summary across all structures. |
| `Probabilities` | Per-class probability maps. |
| `Structures` | Segmented anatomical structures with volume and metadata. |
| `SurfaceDiceScores` | Per-class surface Dice scores at specified tolerance (if ground truth was provided). |
| `UncertaintyMap` | Uncertainty map [H, W] or [D, H, W] indicating model uncertainty per pixel/voxel. |
| `VoxelSpacing` | Voxel spacing in mm [x, y, z] (for 3D results). |

