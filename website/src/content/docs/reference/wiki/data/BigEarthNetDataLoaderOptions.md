---
title: "BigEarthNetDataLoaderOptions"
description: "Configuration options for the BigEarthNet data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the BigEarthNet data loader.

## How It Works

BigEarthNet is a large-scale multi-label remote sensing dataset consisting of 590,326 Sentinel-2
image patches from 10 European countries. Each patch is 120x120 pixels with 12 spectral bands.
The multi-label classification task uses 19 CORINE Land Cover classes (BigEarthNet-19) or
43 original CLC classes.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `NumBands` | Number of spectral bands to load. |
| `Split` | Dataset split to load. |
| `Use19ClassScheme` | Use the simplified 19-class label scheme (BigEarthNet-19). |

