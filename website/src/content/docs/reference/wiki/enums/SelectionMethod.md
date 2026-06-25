---
title: "SelectionMethod"
description: "Methods for selecting individuals for reproduction."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Methods for selecting individuals for reproduction.

## Fields

| Field | Summary |
|:-----|:--------|
| `Rank` | Rank selection - selection probability based on fitness rank rather than absolute value. |
| `RouletteWheel` | Roulette wheel selection - selection probability proportional to fitness. |
| `StochasticUniversalSampling` | Stochastic universal sampling - similar to roulette wheel but with multiple equally spaced pointers. |
| `Tournament` | Tournament selection - randomly select a group of individuals and pick the best. |
| `Truncation` | Truncation selection - select a percentage of the fittest individuals. |
| `Uniform` | Uniform selection - all individuals have an equal chance of being selected. |

