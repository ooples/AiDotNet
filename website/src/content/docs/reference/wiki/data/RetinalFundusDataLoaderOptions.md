---
title: "RetinalFundusDataLoaderOptions"
description: "Configuration options for the Retinal Fundus data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Retinal Fundus data loader.

## How It Works

Retinal fundus photography datasets are used for diabetic retinopathy and glaucoma detection.
This loader supports common retinal datasets such as EyePACS/Kaggle Diabetic Retinopathy
(35K train / 53K test) with 5-class severity grading (0: No DR, 1: Mild, 2: Moderate,
3: Severe, 4: Proliferative DR).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Target image size. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

