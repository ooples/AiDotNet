---
title: "CDNODAlgorithm<T>"
description: "CD-NOD — Constraint-based Discovery from Non-stationary / heterogeneous Data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

CD-NOD — Constraint-based Discovery from Non-stationary / heterogeneous Data.

## For Beginners

CD-NOD exploits the idea that when conditions change, causes
tend to stay the same while effects change. By adding a "time" or "context" variable,
it can detect which relationships shift, giving extra information for causal direction.

## How It Works

CD-NOD extends constraint-based causal discovery to handle data from changing
environments or multiple domains. It augments the data with a context variable C
(representing time index or domain indicator), then uses conditional independence
tests to detect which causal mechanisms change across contexts and leverages these
changes to orient more edges than standard PC.

**Algorithm:**

- Augment data with a context variable C (row index normalized to [0,1])
- Run PC skeleton phase on augmented data (d+1 variables)
- Identify variables connected to C (these have changing distributions)
- For edges where one endpoint changes and the other doesn't:

orient non-changing → changing (cause is stable, effect changes)

- Apply standard v-structure and Meek rules on remaining edges
- Return the d x d adjacency matrix (dropping the context variable)

Reference: Huang et al. (2020), "Causal Discovery from Heterogeneous/Nonstationary Data", JMLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CDNODAlgorithm(CausalDiscoveryOptions)` | Initializes CD-NOD with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsLatentConfounders` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

