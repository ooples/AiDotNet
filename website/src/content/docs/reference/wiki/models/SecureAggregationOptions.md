---
title: "SecureAggregationOptions"
description: "Configuration options for secure aggregation in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for secure aggregation in federated learning.

## How It Works

**For Beginners:** Secure aggregation hides each client's update from the server so the server
can only see the final combined result. These options let you pick how secure aggregation should
behave, including whether the protocol can handle clients dropping out mid-round.

## Properties

| Property | Summary |
|:-----|:--------|
| `Enabled` | Gets or sets whether secure aggregation is enabled. |
| `MaxDropoutFraction` | Gets or sets the maximum fraction of selected clients that may drop out while still completing the round. |
| `MinimumUploaderCount` | Gets or sets the minimum number of clients that must upload masked updates for the round to succeed. |
| `Mode` | Gets or sets which secure aggregation mode is used. |
| `ReconstructionThreshold` | Gets or sets the reconstruction threshold used by dropout-resilient secure aggregation. |

