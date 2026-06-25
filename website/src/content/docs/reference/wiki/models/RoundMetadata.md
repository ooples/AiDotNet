---
title: "RoundMetadata"
description: "Contains detailed metrics for a single federated learning round."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Contains detailed metrics for a single federated learning round.

## How It Works

**For Beginners:** Information about what happened in one specific training round.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageLocalLoss` | Gets or sets the average local loss across selected clients. |
| `ClientContributionScores` | Gets or sets the per-client contribution scores for this round. |
| `CommunicationMB` | Gets or sets the communication cost for this round in megabytes. |
| `ContributionMethodUsed` | Gets or sets the contribution evaluation method used for this round. |
| `DriftDetected` | Gets or sets whether drift was detected in this round. |
| `DriftingClientIds` | Gets or sets the IDs of clients identified as drifting in this round. |
| `GlobalAccuracy` | Gets or sets the global model accuracy after this round. |
| `GlobalLoss` | Gets or sets the global model loss after this round. |
| `PrivacyBudgetConsumed` | Gets or sets the privacy budget consumed in this round. |
| `RoundNumber` | Gets or sets the round number (0-indexed). |
| `RoundTimeSeconds` | Gets or sets the time taken for this round in seconds. |
| `SelectedClientIds` | Gets or sets the IDs of clients selected for this round. |
| `UploadCompressionRatio` | Gets or sets the effective upload compression ratio for this round (1.0 means no compression). |

