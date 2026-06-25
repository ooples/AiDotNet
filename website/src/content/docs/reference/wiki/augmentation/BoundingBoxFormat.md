---
title: "BoundingBoxFormat"
description: "Specifies the format of bounding box coordinates."
section: "API Reference"
---

`Enums` · `AiDotNet.Augmentation.Image`

Specifies the format of bounding box coordinates.

## Fields

| Field | Summary |
|:-----|:--------|
| `COCO` | [x_min, y_min, width, height] - COCO format (same as XYWH). |
| `CXCYWH` | [x_center, y_center, width, height] - Center point with dimensions. |
| `PascalVOC` | [x_min, y_min, x_max, y_max] - Pascal VOC format (same as XYXY). |
| `XYWH` | [x_min, y_min, width, height] - Top-left corner with dimensions. |
| `XYXY` | [x_min, y_min, x_max, y_max] - Absolute pixel coordinates. |
| `YOLO` | [x_center, y_center, width, height] normalized to [0, 1] - YOLO format. |

