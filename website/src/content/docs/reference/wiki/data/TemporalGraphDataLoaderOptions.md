---
title: "TemporalGraphDataLoaderOptions"
description: "Configuration options for the temporal graph data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Graph`

Configuration options for the temporal graph data loader.

## How It Works

Temporal graphs model evolving networks with timestamped edges.
Supports datasets like Wikipedia edits and Reddit posts for dynamic link prediction.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `EdgeFeatureDimension` | Number of edge feature dimensions. |
| `MaxSamples` | Optional maximum number of interactions to load. |
| `NodeFeatureDimension` | Number of node feature dimensions. |
| `Split` | Dataset split to load. |

