---
title: "EuroSatDataLoaderOptions"
description: "Configuration options for the EuroSAT data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the EuroSAT data loader.

## How It Works

EuroSAT is a land use/land cover classification dataset based on Sentinel-2 satellite images.
It contains 27,000 labeled patches (64x64 pixels) across 10 classes: Annual Crop, Forest,
Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, SeaLake.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `Layout` | Axis ordering for the image tensor. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

