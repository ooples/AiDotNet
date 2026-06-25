---
title: "ImageNet1kDataLoaderOptions"
description: "Configuration options for the ImageNet-1K (ILSVRC 2012) data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the ImageNet-1K (ILSVRC 2012) data loader.

## How It Works

ImageNet-1K contains ~1.28M training images and 50K validation images across 1,000 object categories.
Due to its large size (~150GB), auto-download is disabled by default. Provide the data path
to your local copy of the dataset.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size (images are resized to this square dimension). |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

