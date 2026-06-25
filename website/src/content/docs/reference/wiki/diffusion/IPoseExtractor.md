---
title: "IPoseExtractor<T>"
description: "Pluggable keypoint extractor interface."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Diffusion.Preprocessing`

Pluggable keypoint extractor interface. Concrete implementations wrap
pretrained pose-estimation networks (OpenPose / DWPose / RTMPose) and
must return a `[batch, 3, H, W]` tensor whose channels encode the
rendered pose skeleton (or per-keypoint heatmaps stacked as RGB).
Plug into `OpenPosePreprocessor` via its constructor.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractKeypoints(Tensor<>)` | Extracts pose keypoints from an input image batch and renders them as a 3-channel skeleton tensor consumable by ControlNet's pose conditioning branch. |

