---
title: "PascalVocDataLoaderOptions"
description: "Configuration options for the Pascal VOC data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Pascal VOC data loader.

## How It Works

Pascal VOC (Visual Object Classes) is a classic object detection benchmark with 20 categories.
VOC2007 has 5K train/val + 5K test images. VOC2012 has 11.5K train/val images.
Annotations are in XML format (one per image).

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
| `Year` | VOC year version. |

