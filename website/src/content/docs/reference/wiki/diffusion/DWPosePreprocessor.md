---
title: "DWPosePreprocessor<T>"
description: "DWPose whole-body keypoint detection preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

DWPose whole-body keypoint detection preprocessor for ControlNet conditioning.

## For Beginners

DWPose is an improved version of OpenPose that also detects
hand gestures and facial expressions, not just body pose. This gives ControlNet
much finer control over generated human figures.

## How It Works

DWPose (Dual Whole-body Pose) detects body, hand, and face keypoints with improved
accuracy over OpenPose. It produces a more detailed skeleton including finger and
facial landmark detection.

Reference: Yang et al., "Effective Whole-body Pose Estimation with Two-stages Distillation", ICCV 2023

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

