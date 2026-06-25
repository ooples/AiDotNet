---
title: "Ade20kDataLoaderOptions"
description: "Configuration options for the ADE20K semantic segmentation data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the ADE20K semantic segmentation data loader.

## How It Works

ADE20K contains ~25K images with per-pixel semantic annotations across 150 categories.
Standard benchmark for semantic segmentation models.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageHeight` | Image height after resizing. |
| `ImageWidth` | Image width after resizing. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0,1]. |
| `NumClasses` | Number of semantic classes. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

