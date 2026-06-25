---
title: "AnatomicalStructure"
description: "Metadata for a single segmented anatomical structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Metadata for a single segmented anatomical structure.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBox` | Bounding box [x1, y1, x2, y2] or [x1, y1, z1, x2, y2, z2]. |
| `Centroid` | Centroid of the structure in image coordinates (x, y) or (x, y, z). |
| `ClassId` | Class ID. |
| `DiceScore` | Dice score against ground truth (if provided). |
| `HausdorffDistance` | Hausdorff distance in mm against ground truth (if provided). |
| `MeanConfidence` | Mean confidence across all voxels in this structure. |
| `MinConfidence` | Minimum confidence across all voxels (useful for detecting uncertain regions). |
| `Name` | Structure name (e.g., "Liver", "Left Kidney", "Spleen"). |
| `SnomedCode` | SNOMED CT code for this structure (for clinical interoperability). |
| `SurfaceAreaMm2` | Surface area in square millimeters (for 3D). |
| `VolumeOrAreaMm` | Volume in cubic millimeters (for 3D) or area in square millimeters (for 2D). |
| `VoxelCount` | Volume in voxels (for 3D) or area in pixels (for 2D). |

