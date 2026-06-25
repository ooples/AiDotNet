---
title: "ImageNet21kDataLoaderOptions"
description: "Configuration options for the ImageNet-21K data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the ImageNet-21K data loader.

## How It Works

ImageNet-21K contains ~14.2M images across 21,841 categories (the full ImageNet hierarchy).
Due to its massive size, auto-download is disabled by default.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size (images are resized to this square dimension). |
| `MaxClasses` | Number of classes to load. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

