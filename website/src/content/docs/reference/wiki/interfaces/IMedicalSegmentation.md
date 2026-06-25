---
title: "IMedicalSegmentation<T>"
description: "Interface for medical image segmentation models that handle 2D slices and 3D volumetric data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for medical image segmentation models that handle 2D slices and 3D volumetric data.

## For Beginners

Medical segmentation helps doctors by automatically outlining organs,
tumors, and other structures in medical images.

Key differences from regular segmentation:

- Works with 3D volumes (CT/MRI scans are stacks of 2D slices)
- Handles multiple imaging modalities (CT, MRI T1, MRI T2, etc.)
- Requires very high accuracy — mistakes can affect patient care
- Often needs to handle varying image resolutions and orientations

Models implementing this interface:

- nnU-Net v2 (Nature Methods, self-configuring gold standard)
- TransUNet (transformer + U-Net hybrid)
- Swin-UNETR (hierarchical transformer for brain MRI)
- MedSAM / MedSAM 2 (SAM adapted for medical data)
- MedNeXt (ConvNeXt-based, MICCAI 2023)
- UniverSeg (few-shot, no fine-tuning needed)
- BiomedParse (foundation model, 9 imaging modalities)

## How It Works

Medical image segmentation requires special handling for multi-modal imaging data (CT, MRI,
X-ray, ultrasound, pathology), 3D volumetric processing, and clinical safety requirements.
Models must handle varying image resolutions, anisotropic voxel spacing, and multi-class
organ/lesion segmentation with high accuracy.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedModalities` | Gets the imaging modalities this model supports (e.g., CT, MRI_T1, MRI_T2, Xray). |
| `Supports2D` | Gets whether this model supports 2D slice-by-slice segmentation. |
| `Supports3D` | Gets whether this model supports 3D volumetric segmentation. |
| `SupportsFewShot` | Gets whether this model supports few-shot segmentation (generalizing from few examples). |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentFewShot(Tensor<>,Tensor<>,Tensor<>)` | Segments using few-shot examples (for models that support it). |
| `SegmentSlice(Tensor<>)` | Segments a 2D medical image slice. |
| `SegmentVolume(Tensor<>)` | Segments a 3D medical volume. |

