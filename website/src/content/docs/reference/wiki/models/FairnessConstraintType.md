---
title: "FairnessConstraintType"
description: "Specifies the type of fairness constraint to enforce during federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the type of fairness constraint to enforce during federated learning.

## For Beginners

Fairness constraints ensure the global model treats all client groups
equitably. Without constraints, the model may perform well for majority clients but poorly
for underrepresented groups (e.g., a rural hospital in a federation dominated by urban ones).

## Fields

| Field | Summary |
|:-----|:--------|
| `AgnosticFairness` | AFL (Agnostic FL): minimax optimization that is agnostic to test distribution. |
| `DemographicParity` | Demographic parity: model predictions should be independent of group membership. |
| `EqualOpportunity` | Equal opportunity: true positive rate should be equal across groups. |
| `EqualizedOdds` | Equalized odds: true positive and false positive rates should be equal across groups. |
| `FedFair` | FedFair: multi-objective optimization balancing accuracy, fairness, and efficiency via Pareto scalarization. |
| `MinimaxFairness` | Minimax fairness: minimize the worst-case performance across all client groups. |
| `None` | No fairness constraint applied. |
| `QFairFederatedLearning` | q-FFL: parameterized fairness via power-mean. |
| `TiltedERM` | TERM (Tilted ERM): smooth interpolation between average and worst-case optimization using a tilt parameter. |

