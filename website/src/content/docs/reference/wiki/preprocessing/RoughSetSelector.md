---
title: "RoughSetSelector<T>"
description: "Rough Set Theory-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.RoughSet`

Rough Set Theory-based Feature Selection.

## For Beginners

Rough set theory handles uncertainty by grouping
similar objects together. If removing a feature doesn't change which groups
are distinguishable, that feature is redundant. This method finds the
smallest set of features that still tells all objects apart.

## How It Works

Uses rough set theory concepts like positive region and dependency degree
to find the minimal subset of features (reduct) that preserves the
classification ability of the full feature set.

