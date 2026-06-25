---
title: "OpenImagesDataLoaderOptions"
description: "Configuration options for the Open Images V7 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Open Images V7 data loader.

## How It Works

Open Images V7 is a large-scale dataset with ~9M images and bounding box annotations
for 600 object categories. Annotations are provided as CSV files. Due to its massive size,
auto-download is disabled by default.

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

