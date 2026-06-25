---
title: "ZincDataLoaderOptions"
description: "Configuration options for the ZINC molecular dataset data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Graph`

Configuration options for the ZINC molecular dataset data loader.

## How It Works

ZINC contains ~250K drug-like molecules from the ZINC database.
Standard benchmark for graph regression on constrained solubility.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of molecules to load. |
| `UseSubset` | Use the 12K subset instead of full 250K. |

