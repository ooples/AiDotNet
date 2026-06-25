---
title: "KnockoffFilter<T>"
description: "Knockoff Filter for high-dimensional feature selection with FDR control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.HighDimensional`

Knockoff Filter for high-dimensional feature selection with FDR control.

## For Beginners

When testing many features, some will look important
just by chance. The Knockoff Filter creates fake versions of each feature and
asks: "Is the real feature more predictive than its fake twin?" This helps avoid
selecting features that only appear important by luck.

## How It Works

The Knockoff Filter creates "knockoff" versions of each feature that mimic the
correlation structure but are conditionally independent of the target. By comparing
real features against their knockoffs, it controls the False Discovery Rate (FDR).

