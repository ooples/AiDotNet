---
title: "PubLayNetDataLoaderOptions"
description: "Configuration options for the PubLayNet document layout analysis data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the PubLayNet document layout analysis data loader.

## How It Works

PubLayNet contains ~360K document images with layout annotations (text, title, list, table, figure).
Standard benchmark for document layout analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageHeight` | Image height after resizing. |
| `ImageWidth` | Image width after resizing. |
| `MaxRegions` | Maximum number of layout regions per image. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `NumClasses` | Number of layout classes (text, title, list, table, figure). |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

