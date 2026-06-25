---
title: "ZincDataLoader<T>"
description: "Thin wrapper around `MolecularDatasetLoader` for the ZINC dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Graph`

Thin wrapper around `MolecularDatasetLoader` for the ZINC dataset.

## How It Works

ZINC contains ~250K drug-like molecules from the ZINC database.
Standard benchmark for graph regression on constrained solubility.
The 12K subset is commonly used for benchmarking GNNs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZincDataLoader(ZincDataLoaderOptions)` | Initializes a new ZINC data loader. |

