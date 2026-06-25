---
title: "LinkageMethod"
description: "Linkage methods for hierarchical clustering."
section: "API Reference"
---

`Enums` · `AiDotNet.Clustering.Options`

Linkage methods for hierarchical clustering.

## How It Works

The linkage method determines how the distance between two clusters is computed.
Different methods lead to different cluster shapes and behaviors.

## Fields

| Field | Summary |
|:-----|:--------|
| `Average` | Average linkage (UPGMA). |
| `Centroid` | Centroid linkage (UPGMC). |
| `Complete` | Complete linkage (farthest neighbor). |
| `Median` | Median linkage (WPGMC). |
| `Single` | Single linkage (nearest neighbor). |
| `Ward` | Ward's minimum variance method. |
| `Weighted` | Weighted average linkage (WPGMA). |

