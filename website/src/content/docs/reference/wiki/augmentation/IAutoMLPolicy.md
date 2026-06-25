---
title: "IAutoMLPolicy<T, TData>"
description: "Interface for policies that expose their complete search space."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Interface for policies that expose their complete search space.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFromConfigurations(IList<SampledConfiguration>)` | Creates a new policy from sampled configurations. |
| `GetPolicySearchSpace` | Gets the complete policy search space. |
| `SampleConfiguration(Random)` | Samples a random configuration from the search space. |

