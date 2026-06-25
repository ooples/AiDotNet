---
title: "AntColonyFS<T>"
description: "Ant Colony Optimization for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Ant Colony Optimization for feature selection.

## For Beginners

Imagine ants searching for food. When an ant finds a good path,
it leaves a scent (pheromone) that attracts other ants. Over time, better paths get
stronger scents. Here, "paths" are feature subsets, and "food quality" is model performance.
Features that are part of good subsets accumulate more pheromone and are more likely to be selected.

## How It Works

Ant Colony Optimization (ACO) is a swarm intelligence algorithm inspired by how ants
find optimal paths using pheromone trails. In feature selection, ants construct feature
subsets, and pheromone is deposited on good feature combinations.

