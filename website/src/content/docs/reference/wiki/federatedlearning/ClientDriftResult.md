---
title: "ClientDriftResult"
description: "Per-client drift analysis result."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.DriftDetection`

Per-client drift analysis result.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientId` | Gets or sets the client ID. |
| `DriftScore` | Gets or sets the drift score for this client. |
| `DriftStartRound` | Gets or sets the round at which drift was first detected for this client. |
| `DriftType` | Gets or sets the classified drift type. |
| `RecommendedAction` | Gets or sets the recommended action. |
| `SuggestedWeightMultiplier` | Gets or sets the suggested aggregation weight multiplier for this client. |

