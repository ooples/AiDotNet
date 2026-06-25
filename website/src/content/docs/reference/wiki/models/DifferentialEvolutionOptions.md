---
title: "DifferentialEvolutionOptions<T, TInput, TOutput>"
description: "Configuration options for Differential Evolution optimization, a powerful variant of genetic algorithms that is particularly effective for continuous optimization problems."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Differential Evolution optimization, a powerful variant of genetic algorithms
that is particularly effective for continuous optimization problems.

## For Beginners

Differential Evolution is like a more sophisticated version of genetic algorithms.
Instead of random mutations, it creates new solutions by calculating the difference between existing solutions
and using that difference to guide the search. Think of it as learning from the "distance" between good solutions
to find even better ones. It's particularly good at finding optimal values for problems with continuous variables
(like finding the best temperature, pressure, or other numeric values).

## How It Works

Differential Evolution is a population-based optimization algorithm that uses vector differences for
mutation operations. It's known for its robustness, simplicity, and effectiveness in solving complex
optimization problems, especially those with real-valued parameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `CrossoverRate` | Gets or sets the crossover rate for Differential Evolution. |
| `MutationRate` | Gets or sets the mutation rate (also known as the differential weight or F) for Differential Evolution. |
| `PopulationSize` | Gets or sets the size of the population for Differential Evolution. |

