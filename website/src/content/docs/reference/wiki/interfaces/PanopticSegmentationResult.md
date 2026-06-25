---
title: "PanopticSegmentationResult<T>"
description: "Result of panoptic segmentation containing both semantic and instance information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of panoptic segmentation containing both semantic and instance information.

## Properties

| Property | Summary |
|:-----|:--------|
| `InstanceMap` | Per-pixel instance IDs [H, W]. |
| `PanopticMap` | Combined panoptic ID map [H, W] encoded as classId * maxInstances + instanceId. |
| `Segments` | List of detected thing segments with metadata. |
| `SemanticMap` | Per-pixel semantic class labels [H, W]. |

