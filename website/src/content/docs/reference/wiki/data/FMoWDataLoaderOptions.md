---
title: "FMoWDataLoaderOptions"
description: "Configuration options for the Functional Map of the World (fMoW) data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Functional Map of the World (fMoW) data loader.

## How It Works

fMoW is a satellite imagery dataset for temporal land use classification with 62 categories.
It contains over 1M images from 200+ countries with temporal metadata (images of the same
location taken at different times). This makes it useful for temporal analysis of land use change.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

