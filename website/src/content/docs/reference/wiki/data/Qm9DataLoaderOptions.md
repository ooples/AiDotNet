---
title: "Qm9DataLoaderOptions"
description: "Configuration options for the QM9 molecular property prediction data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Graph`

Configuration options for the QM9 molecular property prediction data loader.

## How It Works

QM9 contains ~134K small organic molecules with up to 9 heavy atoms (C, H, N, O, F).
19 quantum mechanical properties computed with DFT. Standard benchmark for molecular GNNs.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of molecules to load. |
| `TargetProperty` | Target property index (0-18) for regression. |

