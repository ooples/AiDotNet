---
title: "PCAlgorithmSelector<T>"
description: "PC Algorithm-based Feature Selection for causal discovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Causal`

PC Algorithm-based Feature Selection for causal discovery.

## For Beginners

The PC algorithm tries to figure out what
causes what by testing if variables become independent when controlling
for other variables. Features that are directly connected to the target
in this causal web are the most important for understanding and prediction.

## How It Works

Uses the PC (Peter-Clark) algorithm to learn a causal graph structure
from data, then selects features that are directly connected to the
target variable in the causal graph.

