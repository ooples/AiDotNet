---
title: "ClientSelectionOptions"
description: "Configuration options for client selection in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for client selection in federated learning.

## How It Works

**For Beginners:** Client selection controls which devices/organizations participate in each round.
In real deployments, many clients may be offline or slow, so selecting a subset per round is common.

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailabilityThreshold` | Gets or sets the minimum availability probability required for a client to be considered available. |
| `ClientAvailabilityProbabilities` | Gets or sets an optional mapping from client ID to an availability probability (0.0 to 1.0). |
| `ClientGroupKeys` | Gets or sets an optional mapping from client ID to a group key for stratified sampling. |
| `ClusterCount` | Gets or sets the number of clusters to use for cluster-based sampling. |
| `ExplorationRate` | Gets or sets the exploration probability for performance-aware sampling (0.0 to 1.0). |
| `KMeansIterations` | Gets or sets the number of k-means iterations for cluster-based sampling. |
| `Strategy` | Gets or sets the selection strategy. |

