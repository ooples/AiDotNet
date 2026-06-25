---
title: "ExhaustiveSearch<T>"
description: "Exhaustive Search Feature Selection (for small feature sets)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Exhaustive Search Feature Selection (for small feature sets).

## For Beginners

This is like trying every possible combination
of ingredients to find the perfect recipe. It's guaranteed to find the best
answer, but becomes impractical with many features because the number of
combinations grows exponentially (2^n for n features).

## How It Works

Exhaustive search evaluates all possible feature subsets to find the optimal
combination. This guarantees finding the best subset but is only feasible
for small numbers of features (typically less than 20).

