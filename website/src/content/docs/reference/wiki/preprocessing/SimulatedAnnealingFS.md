---
title: "SimulatedAnnealingFS<T>"
description: "Simulated Annealing for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Simulated Annealing for feature selection.

## For Beginners

Like heating metal and slowly cooling it to remove defects,
SA starts "hot" (accepting almost any change) and gradually cools (becoming pickier).
Early on, it accepts worse solutions to explore widely. Later, it focuses on refining
the best solutions found. This helps avoid getting stuck in suboptimal solutions.

## How It Works

Simulated Annealing (SA) is inspired by the annealing process in metallurgy.
It explores the search space by accepting worse solutions with decreasing probability
as the "temperature" cools, helping escape local optima.

