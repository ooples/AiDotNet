---
title: "PermutationIndividual"
description: "Represents an individual encoded as a permutation, suitable for problems like TSP."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Genetics`

Represents an individual encoded as a permutation, suitable for problems like TSP.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PermutationIndividual(ICollection<PermutationGene>)` | Creates a permutation individual with the specified genes. |
| `PermutationIndividual(Int32,Random)` | Creates a new permutation individual with a random permutation of the specified size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FillFromOther(Int32[],Int32[],Int32,Int32)` | Helper method for Order Crossover. |
| `GetPermutation` | Gets the permutation as an array of indices. |
| `InversionMutation(Random)` | Applies the inversion mutation by reversing a random subsequence. |
| `OrderCrossover(PermutationIndividual,Random)` | Applies the Order Crossover (OX) operator. |
| `SwapMutation(Random)` | Applies a swap mutation by swapping two random positions. |

