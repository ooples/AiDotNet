---
title: "SegmentationVisualizationConfig"
description: "Visualization settings and utilities for segmentation outputs."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ComputerVision.Segmentation.Common`

Visualization settings and utilities for segmentation outputs.

## For Beginners

After segmenting an image, you usually want to visualize the result.
This class provides configuration for creating color-coded overlay images where each class
or instance gets a distinct color. The overlay can be blended with the original image
to see both the segmentation and the original content.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Alpha blending factor for overlay (0 = fully transparent, 1 = fully opaque). |
| `BackgroundColor` | Background color for areas with no segmentation (R, G, B). |
| `ColorPalette` | Custom color palette as RGB triplets [numColors, 3] with values in [0, 255]. |
| `ContourThickness` | Contour thickness in pixels. |
| `DrawContours` | Whether to draw contours/boundaries around segmented regions. |
| `MinDisplayConfidence` | Minimum confidence threshold for displaying instances. |
| `ShowBoundingBoxes` | Whether to display instance bounding boxes. |
| `ShowLabels` | Whether to display class labels on the visualization. |
| `ShowScores` | Whether to display confidence scores alongside labels. |
| `UseFixedPalette` | Whether to use a fixed color palette or generate random colors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetADE20KPalette` | Gets the default ADE20K color palette (150 classes). |
| `GetCOCOPalette` | Gets the default COCO panoptic color palette (133 classes). |
| `GetCityscapesPalette` | Gets a Cityscapes-style color palette (19 classes). |

