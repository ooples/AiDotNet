---
title: "BudgetConstrainedFS<T>"
description: "Budget-Constrained Feature Selection with a maximum cost budget."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.CostSensitive`

Budget-Constrained Feature Selection with a maximum cost budget.

## For Beginners

Imagine you have $100 to spend on data collection.
This selector picks the most valuable features you can afford within your budget.
It's like a knapsack problem - get the most value without exceeding your limit.

## How It Works

Budget-Constrained Feature Selection selects the most informative features
while staying within a specified cost budget. It uses a greedy approach
to maximize value per unit cost.

