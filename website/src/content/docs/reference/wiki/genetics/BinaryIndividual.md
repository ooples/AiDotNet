---
title: "BinaryIndividual"
description: "Represents an individual encoded with binary genes, suitable for classic GA problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Genetics`

Represents an individual encoded with binary genes, suitable for classic GA problems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BinaryIndividual(ICollection<BinaryGene>)` | Creates a binary individual with the specified genes. |
| `BinaryIndividual(Int32,Random)` | Creates a new binary individual with the specified chromosome length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetValueAsInt` | Gets the binary value of this individual as an integer. |
| `GetValueAsNormalizedDouble` | Gets the binary value of this individual as a double in the range [0,1]. |
| `GetValueMapped(Double,Double)` | Maps the binary string to a double value within the specified range. |

