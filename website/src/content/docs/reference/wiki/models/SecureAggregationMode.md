---
title: "SecureAggregationMode"
description: "Determines which secure aggregation protocol variant is used."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Determines which secure aggregation protocol variant is used.

## How It Works

**For Beginners:** Secure aggregation comes in different "flavors" depending on how
much client drop-out the protocol can tolerate.

## Fields

| Field | Summary |
|:-----|:--------|
| `FullParticipation` | Synchronous secure aggregation that requires full participation from the selected clients. |
| `ThresholdDropoutResilient` | Dropout-resilient secure aggregation with a reconstruction threshold. |

