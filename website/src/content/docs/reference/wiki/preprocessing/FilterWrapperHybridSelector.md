---
title: "FilterWrapperHybridSelector<T>"
description: "Filter-Wrapper Hybrid Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Filter-Wrapper Hybrid Feature Selection.

## For Beginners

This method gets the best of both worlds.
First, it quickly eliminates obviously irrelevant features using simple
statistics (filter). Then, it carefully evaluates the remaining features
using model performance (wrapper). This is faster than pure wrapper methods
but more accurate than pure filter methods.

## How It Works

Combines filter and wrapper methods: first uses a fast filter to reduce
the feature space, then applies a more accurate wrapper method on the
reduced set.

