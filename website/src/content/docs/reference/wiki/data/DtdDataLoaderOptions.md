---
title: "DtdDataLoaderOptions"
description: "Configuration options for the Describable Textures Dataset (DTD) loader (Cimpoi et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Describable Textures Dataset (DTD) loader (Cimpoi et al. 2014).

## How It Works

DTD — 47 texture classes × 120 images each (5,640 total). The canonical
texture-classification benchmark. Ten predefined train/val/test splits;
this loader uses split #1 by default (matches the standard reporting
convention). Image-list-based labels via `labels/{train,val,test}1.txt`.

## Properties

| Property | Summary |
|:-----|:--------|
| `SplitIndex` | Predefined split index (1..10). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

