---
title: "GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for genetic algorithm optimization, which uses principles inspired by natural selection to find optimal solutions to complex problems."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for genetic algorithm optimization, which uses principles inspired by natural selection
to find optimal solutions to complex problems.

## For Beginners

A genetic algorithm works like breeding animals or plants to get desired traits.
You start with a diverse group of potential solutions (the "population"), evaluate how good each one is,
let the best ones "reproduce" by combining their characteristics, occasionally introduce random changes
("mutations"), and repeat this process over multiple "generations" until you find an excellent solution.
It's a way to solve problems by mimicking how nature evolves species over time.

## How It Works

Genetic algorithms simulate the process of natural selection where the fittest individuals are selected for
reproduction to produce offspring for the next generation. This approach is particularly effective for
optimization problems with large search spaces or complex constraints.

## Properties

| Property | Summary |
|:-----|:--------|
| `CrossoverRate` | Gets or sets the rate at which solutions are combined to create new ones in genetic algorithms. |
| `CrossoverRateDecay` | Gets or sets the factor by which the crossover rate decreases when progress stalls or reverses. |
| `CrossoverRateIncrease` | Gets or sets the factor by which the crossover rate increases when progress is being made. |
| `MaxCrossoverRate` | Gets or sets the maximum allowed crossover rate for genetic algorithms. |
| `MaxGenerations` | Gets or sets the maximum number of generations (iterations) the genetic algorithm will run. |
| `MaxMutationRate` | Gets or sets the maximum allowed mutation rate for genetic algorithms. |
| `MaxPopulationSize` | Gets or sets the maximum allowed population size for genetic algorithms. |
| `MinCrossoverRate` | Gets or sets the minimum allowed crossover rate for genetic algorithms. |
| `MinMutationRate` | Gets or sets the minimum allowed mutation rate for genetic algorithms. |
| `MinPopulationSize` | Gets or sets the minimum allowed population size for genetic algorithms. |
| `MutationRate` | Gets or sets the mutation rate for genetic and evolutionary algorithms. |
| `MutationRateDecay` | Gets or sets the factor by which the mutation rate decreases when progress is being made. |
| `MutationRateIncrease` | Gets or sets the factor by which the mutation rate increases when progress stalls or reverses. |
| `PopulationSize` | Gets or sets the size of the population for genetic and evolutionary algorithms. |

