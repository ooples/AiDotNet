---
title: "VflEpochResult<T>"
description: "Contains metrics from a single VFL training epoch."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Contains metrics from a single VFL training epoch.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageLoss` | Gets or sets the average training loss for this epoch. |
| `BatchesProcessed` | Gets or sets the number of batches processed in this epoch. |
| `Epoch` | Gets or sets the epoch number (0-indexed). |
| `EpochTime` | Gets or sets the time taken for this epoch. |
| `PrivacyBudgetSpent` | Gets or sets the cumulative privacy budget spent (epsilon, delta) if label DP is enabled. |
| `SamplesProcessed` | Gets or sets the number of samples processed in this epoch. |

