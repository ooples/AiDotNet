---
title: "MinimalRedundancySelector<T>"
description: "Minimal Redundancy Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Redundancy`

Minimal Redundancy Feature Selection.

## For Beginners

When features are highly correlated with each
other, they provide similar information. This selector picks features that
are different from each other while still being useful for prediction.

## How It Works

Selects features that minimize redundancy among selected features while
maximizing relevance to the target (mRMR-like approach).

