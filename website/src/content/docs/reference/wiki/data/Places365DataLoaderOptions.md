---
title: "Places365DataLoaderOptions"
description: "Configuration options for the Places365 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Places365 data loader.

## How It Works

Places365 is a scene recognition dataset with 1.8M training images across 365 scene categories
(e.g., bedroom, kitchen, forest, highway). Available in standard (256x256) and challenge (high-res) versions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

