---
title: "StackingFeatureSelector<T>"
description: "Stacking-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Stacking-based Feature Selection.

## For Beginners

Stacking combines multiple ways of measuring
feature importance (like correlation, variance, and mutual information).
A second layer then learns how to best combine these measures, giving you
more reliable feature selection than any single method alone.

## How It Works

Uses a stacking ensemble approach where multiple base feature scorers are
combined using a meta-learner to produce final feature rankings.

