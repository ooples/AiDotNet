---
title: "GeneticParameters"
description: "Parameters for configuring a genetic algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Parameters for configuring a genetic algorithm.

## Properties

| Property | Summary |
|:-----|:--------|
| `CrossoverOperator` | Gets or sets the name of the crossover operator to use. |
| `CrossoverRate` | Gets or sets the probability of crossover occurring. |
| `ElitismRate` | Gets or sets the elitism rate (percentage of top individuals to preserve unchanged). |
| `FitnessThreshold` | Gets or sets the fitness threshold for termination. |
| `InitializationMethod` | Gets or sets the initialization method to use for creating the initial population. |
| `MaxGenerations` | Gets or sets the maximum number of generations to evolve. |
| `MaxGenerationsWithoutImprovement` | Gets or sets the maximum number of generations without improvement before termination. |
| `MaxTime` | Gets or sets the maximum time allowed for evolution. |
| `MutationOperator` | Gets or sets the name of the mutation operator to use. |
| `MutationRate` | Gets or sets the probability of mutation occurring. |
| `PopulationSize` | Gets or sets the size of the population. |
| `SelectionMethod` | Gets or sets the selection method to use. |
| `TournamentSize` | Gets or sets the tournament size for tournament selection. |
| `UseParallelEvaluation` | Gets or sets whether to use parallel evaluation of fitness. |

