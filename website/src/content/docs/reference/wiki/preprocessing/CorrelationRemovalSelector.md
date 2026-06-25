---
title: "CorrelationRemovalSelector<T>"
description: "Correlation-based Feature Removal."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Correlation-based Feature Removal.

## For Beginners

When two features are highly correlated, they
provide similar information. This method finds pairs of highly correlated
features and removes one from each pair, keeping your dataset compact
without losing much information.

## How It Works

Removes highly correlated features to reduce multicollinearity, keeping
one feature from each group of highly correlated features.

