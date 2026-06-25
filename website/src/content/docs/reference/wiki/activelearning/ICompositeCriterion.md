---
title: "ICompositeCriterion<T>"
description: "Interface for composite stopping criteria (multiple criteria combined)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for composite stopping criteria (multiple criteria combined).

## Properties

| Property | Summary |
|:-----|:--------|
| `Criteria` | Gets the individual criteria in this composite. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCriterion(IStoppingCriterion<>)` | Adds a criterion to the composite. |
| `RemoveCriterion(IStoppingCriterion<>)` | Removes a criterion from the composite. |

