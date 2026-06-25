---
title: "SemanticKittiDataLoader<T>"
description: "Loads the SemanticKITTI dataset (per-point semantic labels for LiDAR point clouds)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Geometry`

Loads the SemanticKITTI dataset (per-point semantic labels for LiDAR point clouds).

## How It Works

SemanticKITTI expects:

Features are point cloud Tensor[N, PointsPerSample * 3].
Labels are per-point semantic class Tensor[N, PointsPerSample].

