---
title: "RealValuedIndividual"
description: "Represents an individual encoded with real-valued genes, suitable for numerical optimization problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Genetics`

Represents an individual encoded with real-valued genes, suitable for numerical optimization problems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealValuedIndividual(ICollection<RealGene>)` | Creates a real-valued individual with the specified genes. |
| `RealValuedIndividual(Int32,Double,Double,Random)` | Creates a new individual with random values within the specified range. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetValuesAsArray` | Gets the values of all genes as an array. |
| `UpdateStepSizes(Double)` | Updates the step sizes according to Evolutionary Strategies 1/5 success rule. |

