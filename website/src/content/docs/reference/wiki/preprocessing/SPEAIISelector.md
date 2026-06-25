---
title: "SPEAIISelector<T>"
description: "SPEA-II Multi-objective Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.MultiObjective`

SPEA-II Multi-objective Feature Selection.

## For Beginners

SPEA-II keeps track of the best solutions found
so far in an "archive." It measures how good a solution is by counting how
many other solutions it dominates (is better than in all objectives).

## How It Works

Uses the Strength Pareto Evolutionary Algorithm II (SPEA-II) which maintains
an archive of non-dominated solutions and uses fine-grained fitness assignment.

