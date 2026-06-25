---
title: "CocoDetectionDataLoaderOptions"
description: "Configuration options for the COCO Detection data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the COCO Detection data loader.

## How It Works

COCO (Common Objects in Context) 2017 Detection contains 118K training and 5K validation images
with 80 object categories. Annotations include bounding boxes, segmentation masks, and captions.
This loader focuses on the object detection task (bounding boxes + class labels).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxDetections` | Maximum number of detections per image. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

