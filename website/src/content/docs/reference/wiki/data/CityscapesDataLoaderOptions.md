---
title: "CityscapesDataLoaderOptions"
description: "Configuration options for the Cityscapes semantic segmentation loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Cityscapes semantic segmentation loader.

## How It Works

Cityscapes is the canonical urban-driving semantic segmentation
benchmark (Cordts et al. 2016). The "fine" annotations have 5,000
pixel-accurate labelled images at 2048×1024 across 50 cities. 19
evaluation classes (out of 30 source classes; 11 are reserved/ignored).

**Auto-download is disabled by default** — Cityscapes requires
account sign-up at cityscapes-dataset.com. Download the two archives
manually and extract under `DataPath`:
`leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Auto-download is OFF by default — Cityscapes requires manual sign-up. |
| `MapToTrainIds` | Map the 30 source classes to the 19 evaluation classes (CityscapesScripts ID2trainID). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

