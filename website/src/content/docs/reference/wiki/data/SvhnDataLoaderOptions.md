---
title: "SvhnDataLoaderOptions"
description: "Configuration options for the SVHN (Street View House Numbers) loader (Netzer et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the SVHN (Street View House Numbers) loader (Netzer et al. 2011).

## How It Works

SVHN Format-2 (cropped digits) — 32×32 RGB digit classification, 73,257
train + 26,032 test + 531,131 extra. The "harder than MNIST" baseline,
used widely for early CNN comparison studies and SSL ablations.

## Properties

| Property | Summary |
|:-----|:--------|
| `IncludeExtra` | Include the 531k "extra" samples in the train split. |

