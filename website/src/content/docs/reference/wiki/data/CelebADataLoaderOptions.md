---
title: "CelebADataLoaderOptions"
description: "Configuration options for the CelebA face attributes data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the CelebA face attributes data loader.

## How It Works

CelebA contains ~200K celebrity face images with 40 binary attribute annotations.
Standard benchmark for face attribute prediction and generation.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageHeight` | Image height after resizing. |
| `ImageWidth` | Image width after resizing. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0,1]. |
| `NumAttributes` | Number of binary face attributes. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

