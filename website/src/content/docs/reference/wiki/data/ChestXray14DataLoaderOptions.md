---
title: "ChestXray14DataLoaderOptions"
description: "Configuration options for the NIH Chest X-ray 14 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the NIH Chest X-ray 14 data loader.

## How It Works

NIH Chest X-ray 14 contains 112,120 frontal-view chest X-ray images from 30,805 unique patients
with 14 disease labels mined from radiology reports using NLP. This is a multi-label classification
task where each image can have zero or more of the 14 disease labels.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

