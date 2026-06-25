---
title: "DriftReport"
description: "Comprehensive report of drift analysis across all federated clients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.DriftDetection`

Comprehensive report of drift analysis across all federated clients.

## For Beginners

After running drift detection, this report tells you which clients
are experiencing data distribution shifts, how severe they are, and what actions to take.
A global drift flag indicates when enough clients are drifting that the entire federation
should adapt (e.g., increase learning rate, trigger retraining).

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageDriftScore` | Gets or sets the average drift score across all clients. |
| `ClientResults` | Gets or sets per-client drift results. |
| `DriftingClientFraction` | Gets or sets the fraction of clients currently experiencing drift. |
| `GlobalDriftDetected` | Gets or sets whether global drift is detected (enough clients are drifting). |
| `Method` | Gets or sets the detection method used. |
| `Round` | Gets or sets the FL round at which this report was generated. |
| `Summary` | Gets or sets a human-readable summary of the drift analysis. |
| `Timestamp` | Gets or sets the UTC timestamp of the report. |

