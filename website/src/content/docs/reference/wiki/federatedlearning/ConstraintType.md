---
title: "ConstraintType"
description: "Types of constraints that can be proven in zero-knowledge."
section: "API Reference"
---

`Enums` · `AiDotNet.FederatedLearning.Verification`

Types of constraints that can be proven in zero-knowledge.

## Fields

| Field | Summary |
|:-----|:--------|
| `CommitmentOpening` | Prove value equals a committed value (commitment opening). |
| `ElementBound` | Prove each element is in [-Bound, Bound]. |
| `NormBound` | Prove L2 norm is at most Bound. |
| `ScalarBound` | Prove a scalar value is at most Bound. |

