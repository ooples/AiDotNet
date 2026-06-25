---
title: "KnockoffFilter<T>"
description: "Knockoff Filter for false discovery rate control in feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.HighDimensional`

Knockoff Filter for false discovery rate control in feature selection.

## For Beginners

Knockoffs are fake features designed to compete
with real ones. If a real feature beats its knockoff, it's likely important.
This controls how many false positives sneak into your selection.

## How It Works

The Knockoff Filter creates "knockoff" copies of features that mimic their
correlation structure but are independent of the target. Features whose
importance exceeds their knockoffs' importance are selected.

This method provides FDR (False Discovery Rate) control, guaranteeing that
the expected proportion of false discoveries is below a specified threshold.

