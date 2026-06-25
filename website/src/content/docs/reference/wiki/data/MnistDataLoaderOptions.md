---
title: "MnistDataLoaderOptions"
description: "Configuration options for the MNIST data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the MNIST data loader.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download the dataset if not present. |
| `DataPath` | Root data path. |
| `Flatten` | Whether to flatten images to 1D vectors (784) instead of the spatial layout (`[B, 28, 28, 1]` NHWC or `[B, 1, 28, 28]` NCHW). |
| `Layout` | Axis ordering for the image tensor. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Whether to normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

