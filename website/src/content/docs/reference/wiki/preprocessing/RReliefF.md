---
title: "RReliefF<T>"
description: "RReliefF algorithm for regression problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Relief`

RReliefF algorithm for regression problems.

## For Beginners

Standard Relief uses class labels (hit/miss).
RReliefF works with numeric targets by weighting neighbors based on
how similar their target values are to the sample's target.

## How It Works

RReliefF extends ReliefF to handle continuous targets (regression).
It uses distance-weighted contributions based on target value differences.

