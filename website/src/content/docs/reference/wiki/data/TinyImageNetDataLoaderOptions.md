---
title: "TinyImageNetDataLoaderOptions"
description: "Configuration options for the Tiny ImageNet (200-class, 64×64) data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Tiny ImageNet (200-class, 64×64) data loader.

## How It Works

Tiny ImageNet is the standard middle-ground vision benchmark between
CIFAR-100 and full ImageNet — 200 classes with 500 training, 50
validation, 50 test images each at 64×64 resolution. Used widely for
architecture-search and few-shot studies. Produced by Stanford CS231n.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageSize` | Image size (height = width). |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |

