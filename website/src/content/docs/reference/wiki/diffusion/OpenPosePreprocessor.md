---
title: "OpenPosePreprocessor<T>"
description: "OpenPose body keypoint detection preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

OpenPose body keypoint detection preprocessor for ControlNet conditioning.

## For Beginners

This finds people in your image and draws stick-figure skeletons
showing their pose. ControlNet uses this to generate new images with people in the
same positions.

## How It Works

Detects human body keypoints (joints, limbs) and renders them as a pose skeleton
visualization. The output is a 3-channel RGB image showing detected poses.

Reference: Cao et al., "OpenPose: Realtime Multi-Person 2D Pose Estimation", IEEE TPAMI 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenPosePreprocessor(IPoseExtractor<>)` | Constructs the preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

