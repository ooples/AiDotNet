---
title: "ProteinDataLoaderOptions"
description: "Configuration options for the protein structure graph data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Graph`

Configuration options for the protein structure graph data loader.

## How It Works

Loads protein structures as graphs where amino acids are nodes and spatial/sequence
contacts are edges. Supports function prediction and fold classification tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `ContactThreshold` | Contact distance threshold in angstroms. |
| `DataPath` | Root data path. |
| `FeatureDimension` | Number of amino acid feature dimensions. |
| `MaxSamples` | Optional maximum number of proteins to load. |
| `NumClasses` | Number of functional classes. |

