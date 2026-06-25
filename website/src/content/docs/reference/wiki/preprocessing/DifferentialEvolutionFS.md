---
title: "DifferentialEvolutionFS<T>"
description: "Differential Evolution for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Differential Evolution for feature selection.

## For Beginners

DE evolves a population of candidate solutions. To create
new solutions, it takes differences between existing ones and adds them to others.
This creates mutations that explore the search space efficiently. For feature selection,
these continuous values are converted to yes/no decisions about each feature.

## How It Works

Differential Evolution (DE) is an evolutionary optimization algorithm that uses
mutation based on the difference between population members. For feature selection,
real-valued vectors are converted to binary feature masks using thresholding.

