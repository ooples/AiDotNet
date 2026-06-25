---
title: "INaturalistDataLoaderOptions"
description: "Configuration options for the iNaturalist data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the iNaturalist data loader.

## How It Works

iNaturalist is a large-scale species classification dataset with fine-grained categories.
The 2021 version contains ~2.7M images across 10,000 species. The dataset exhibits
significant class imbalance (long-tailed distribution), making it useful for imbalanced learning research.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |
| `Version` | iNaturalist version year. |

