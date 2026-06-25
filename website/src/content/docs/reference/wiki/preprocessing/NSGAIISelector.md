---
title: "NSGAIISelector<T>"
description: "NSGA-II Multi-objective Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.MultiObjective`

NSGA-II Multi-objective Feature Selection.

## For Beginners

Sometimes we want both good accuracy AND few
features. NSGA-II finds multiple solutions that trade off between these goals.
Some solutions have high accuracy but many features; others have fewer
features but slightly lower accuracy. You can then choose the best trade-off.

## How It Works

Uses the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize
multiple objectives simultaneously: maximizing prediction accuracy while
minimizing the number of selected features.

