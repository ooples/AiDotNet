---
title: "SimulatedAnnealingSelector<T>"
description: "Simulated Annealing based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Optimization`

Simulated Annealing based Feature Selection.

## For Beginners

Like cooling metal, this starts with high "temperature"
where it accepts almost any change, then gradually cools down and becomes pickier.
This helps find good feature combinations by exploring broadly first, then refining.

## How It Works

Uses simulated annealing optimization to find optimal feature subsets by
accepting both improvements and occasionally worse solutions to escape local optima.

