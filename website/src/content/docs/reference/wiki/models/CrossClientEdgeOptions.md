---
title: "CrossClientEdgeOptions"
description: "Configuration for handling cross-client edges in federated graph learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for handling cross-client edges in federated graph learning.

## For Beginners

When a graph is split across clients, some edges connect nodes on
different clients. These "cross-client edges" are a unique challenge — we need to discover them
without revealing each client's full adjacency list. This class controls how that discovery works.

## How It Works

**Options:**

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheDiscoveredEdges` | Gets or sets whether to cache discovered cross-client edges across rounds. |
| `EdgePrivacyEpsilon` | Gets or sets the differential privacy epsilon for edge queries. |
| `MaxEdgesPerClientPair` | Gets or sets the maximum number of cross-client edges to discover per client pair. |
| `UsePsi` | Gets or sets whether to use PSI for edge discovery. |
| `UseTee` | Gets or sets whether to use TEE-based edge discovery as an alternative to PSI. |

