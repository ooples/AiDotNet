---
title: "FederatedFairnessOptions"
description: "Configuration options for fairness constraints in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for fairness constraints in federated learning.

## For Beginners

These options control how the federated learning system ensures fair
treatment of different client groups. For example, if your federation includes hospitals from both
wealthy and underserved areas, fairness constraints prevent the model from favoring wealthier
hospitals at the expense of underserved ones.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConstraintType` | Gets or sets the type of fairness constraint to enforce. |
| `Enabled` | Gets or sets whether fairness constraints are enabled. |
| `EvaluationFrequency` | Gets or sets how often to evaluate fairness metrics (in rounds). |
| `FairnessLambda` | Gets or sets the weight of the fairness penalty in the aggregation objective. |
| `FairnessThreshold` | Gets or sets the maximum allowed fairness violation before corrective action. |
| `MinimaxBoostFactor` | Gets or sets the boost factor for underperforming groups in minimax fairness. |
| `NumberOfGroups` | Gets or sets the number of client groups for group-based fairness constraints. |

