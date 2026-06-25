---
title: "SkeletonDefinition"
description: "Represents a skeleton definition for pose estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Represents a skeleton definition for pose estimation.

## How It Works

A skeleton defines the structure of keypoints and their connections.

## Properties

| Property | Summary |
|:-----|:--------|
| `Connections` | Gets or sets the skeleton connections as pairs of keypoint indices. |
| `KeypointNames` | Gets or sets the keypoint names in order. |
| `KeypointOKSConstants` | Gets or sets the per-keypoint OKS constants (for COCO-style evaluation). |
| `SymmetricPairs` | Gets or sets the left-right symmetric keypoint pairs for horizontal flip. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateCOCO` | Creates a standard COCO human pose skeleton with 17 keypoints. |
| `CreateMPII` | Creates a MPII human pose skeleton with 16 keypoints. |

