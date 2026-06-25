---
title: "SkinLesionDataLoaderOptions"
description: "Configuration options for the ISIC Skin Lesion data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the ISIC Skin Lesion data loader.

## How It Works

The ISIC (International Skin Imaging Collaboration) Skin Lesion dataset is used for
dermoscopic image classification. The 2019 challenge version contains ~25K training images
across 8 diagnostic categories (e.g., melanoma, basal cell carcinoma, dermatofibroma).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

