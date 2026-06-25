---
title: "GES_Selector<T>"
description: "Greedy Equivalence Search (GES) Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Causal`

Greedy Equivalence Search (GES) Feature Selection.

## For Beginners

GES finds the best causal structure by
greedily adding and removing edges to maximize a score (like BIC).
Features connected to the target in the learned structure are selected.

## How It Works

Uses the GES algorithm for causal structure learning, which searches
over equivalence classes of DAGs using a score-based approach.

