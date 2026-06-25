---
title: "OxfordPetsDataLoaderOptions"
description: "Configuration options for the Oxford-IIIT Pet dataset loader (Parkhi et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Oxford-IIIT Pet dataset loader (Parkhi et al. 2012).

## How It Works

Oxford-IIIT Pets — 37 dog/cat breeds, ~200 images per breed (7,349 total).
Standard fine-grained classification benchmark with both species (binary)
and breed (37-way) labels. Filenames encode breeds: e.g.
`Abyssinian_100.jpg`.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target square image edge in pixels. |
| `MaxSamples` | Optional maximum number of samples to load (for fast iteration / smoke testing). |
| `Normalize` | Normalize byte pixel values to [0, 1] when true (default), or keep raw 0..255 when false. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

