---
title: "ICommitteeStrategy<T, TInput, TOutput>"
description: "Interface for committee-based query strategies (Query By Committee)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for committee-based query strategies (Query By Committee).

## Properties

| Property | Summary |
|:-----|:--------|
| `Committee` | Gets the committee of models. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDisagreement()` | Computes the disagreement among committee members for a sample. |
| `UpdateCommittee(IDataset<,,>)` | Updates all committee members with new training data. |

