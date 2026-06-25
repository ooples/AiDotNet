---
title: "FederatedDriftOptions"
description: "Configuration options for federated concept drift detection and adaptation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated concept drift detection and adaptation.

## For Beginners

In production, client data changes over time (e.g., fraud patterns
evolve, user behavior shifts seasonally). These options configure how the FL system detects
when client data has drifted from the training distribution and how it adapts.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptAggregationWeights` | Gets or sets whether to automatically adapt aggregation weights based on drift scores. |
| `DetectionFrequency` | Gets or sets how often to run drift detection (in rounds). |
| `Enabled` | Gets or sets whether drift detection is enabled. |
| `GlobalDriftThreshold` | Gets or sets the global drift threshold above which all clients are notified. |
| `LookbackWindowRounds` | Gets or sets the number of recent rounds to consider for drift analysis. |
| `Method` | Gets or sets the drift detection method. |
| `MinDriftWeight` | Gets or sets the minimum weight multiplier for heavily drifting clients. |
| `SensitivityThreshold` | Gets or sets the sensitivity threshold for drift detection. |
| `TriggerSelectiveRetraining` | Gets or sets whether to trigger selective retraining when global drift is detected. |

