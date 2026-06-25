---
title: "StanfordCarsDataLoaderOptions"
description: "Configuration for the Stanford Cars dataset loader (Krause et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration for the Stanford Cars dataset loader (Krause et al. 2013).

## How It Works

Stanford Cars — 196 fine-grained car-model classes, 16,185 images.
Standard fine-grained classification benchmark. Original Stanford URLs
have been intermittent over the years, so `AutoDownload`
defaults to false; download manually from the various community mirrors
or HuggingFace and extract under `DataPath`.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Auto-download is OFF by default — Stanford URLs are unstable. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

