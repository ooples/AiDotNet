---
title: "IFeatureAware"
description: "Interface for models that can provide information about their feature usage."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for models that can provide information about their feature usage.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used by this model. |
| `IsFeatureUsed(Int32)` | Checks if a specific feature is used by this model. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |

