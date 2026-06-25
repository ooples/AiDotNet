---
title: "GeneticAlgorithmFS<T>"
description: "Genetic Algorithm for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Genetic Algorithm for feature selection.

## For Beginners

Think of it as natural selection for features. Each
individual is a binary string (1=include feature, 0=exclude). The fittest individuals
(best feature subsets) "breed" to create new solutions. Over generations, the
population evolves toward better feature combinations.

## How It Works

Genetic Algorithms (GA) evolve a population of candidate feature subsets using
biological-inspired operators: selection, crossover, and mutation. Better solutions
are more likely to survive and produce offspring, gradually improving the population.

