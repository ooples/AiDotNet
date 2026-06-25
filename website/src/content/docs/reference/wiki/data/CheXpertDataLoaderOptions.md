---
title: "CheXpertDataLoaderOptions"
description: "Configuration options for the CheXpert data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the CheXpert data loader.

## How It Works

CheXpert (Chest eXpert) is a large chest radiograph dataset from Stanford with 224,316 chest X-rays
of 65,240 patients. It has 14 observation labels with explicit uncertainty handling:
positive (1), negative (0), uncertain (-1), or not mentioned (blank).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |
| `UncertaintyHandling` | Policy for handling uncertain labels (-1). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

