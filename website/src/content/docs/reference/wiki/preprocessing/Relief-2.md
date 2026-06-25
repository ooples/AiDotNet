---
title: "Relief<T>"
description: "Relief algorithm for feature selection based on instance-based learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Relief algorithm for feature selection based on instance-based learning.

## For Beginners

Relief works by looking at individual examples.
For each example, it finds the nearest example from the same class (hit) and
the nearest from a different class (miss). Good features should have similar
values for hits and different values for misses. This intuitive approach
works well for detecting local feature relevance.

## How It Works

Relief estimates feature quality by sampling instances and computing the
difference between distances to nearest hits (same class) and nearest misses
(different class). Features that differentiate classes get higher weights.

