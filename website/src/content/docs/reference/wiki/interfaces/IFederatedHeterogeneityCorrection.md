---
title: "IFederatedHeterogeneityCorrection<T>"
description: "Applies a heterogeneity correction transform to client updates in federated learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Applies a heterogeneity correction transform to client updates in federated learning.

## How It Works

**For Beginners:** Some federated algorithms change how client updates are interpreted before the server aggregates them.
This interface lets AiDotNet swap in different correction methods while keeping the public facade simple.

## Methods

| Method | Summary |
|:-----|:--------|
| `Correct(Int32,Int32,Vector<>,Vector<>,Int32)` | Returns corrected client parameters to be used for aggregation. |
| `GetCorrectionName` | Gets the name of the correction method. |

