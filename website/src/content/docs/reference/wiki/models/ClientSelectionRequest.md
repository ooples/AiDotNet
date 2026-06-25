---
title: "ClientSelectionRequest"
description: "Represents a request to select participating clients for a federated learning round."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a request to select participating clients for a federated learning round.

## How It Works

**For Beginners:** This object packages up the information needed to decide which clients should
join the next training round (who is available, how many to pick, and any optional hints like groups).

## Properties

| Property | Summary |
|:-----|:--------|
| `CandidateClientIds` | Gets or sets the full set of candidate client IDs. |
| `ClientAvailabilityProbabilities` | Gets or sets optional client availability probabilities (0.0 to 1.0). |
| `ClientEmbeddings` | Gets or sets optional per-client embeddings for cluster-based selection. |
| `ClientGroupKeys` | Gets or sets optional group keys for stratified selection. |
| `ClientPerformanceScores` | Gets or sets optional per-client performance scores (higher is better). |
| `ClientWeights` | Gets or sets client weights (typically proportional to sample count). |
| `FractionToSelect` | Gets or sets the fraction of clients to select (0.0 to 1.0). |
| `Random` | Gets or sets the random number generator to use for selection. |
| `RoundNumber` | Gets or sets the round number (0-based) for which selection is being performed. |

