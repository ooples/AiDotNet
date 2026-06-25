---
title: "FastIAMBAlgorithm<T>"
description: "Fast-IAMB — faster variant of IAMB using speculative forward selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

Fast-IAMB — faster variant of IAMB using speculative forward selection.

## For Beginners

Fast-IAMB is like IAMB but more aggressive — instead of
adding one variable at a time, it adds ALL relevant variables at once. This is
much faster for high-dimensional data but might add some false positives. The
backward phase then cleans up by removing variables that were added incorrectly.

## How It Works

Fast-IAMB accelerates IAMB by adding multiple variables at once in the forward phase
(speculative addition), then relying on the backward phase to remove false positives.
In each forward step, ALL variables with significant association are added simultaneously
rather than just the single best one.

**Algorithm:**

- **Speculative forward phase:** In each round, test ALL remaining variables

for association with the target given the current blanket. Add ALL that are
significantly associated (not just the single best)

- **Backward phase:** For each member of the blanket, test if it becomes

conditionally independent of the target given the remaining blanket members.
Remove any that become independent

- Build skeleton and orient edges (same as IAMB)

Reference: Yaramakala and Margaritis (2005), "Speculative Markov Blanket Discovery
for Optimal Feature Selection".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastIAMBAlgorithm(CausalDiscoveryOptions)` | Initializes Fast-IAMB with optional configuration. |

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
| `FindMarkovBlanketFast(Matrix<>,Int32,Int32)` | Finds the Markov blanket using speculative (batch) forward selection. |

