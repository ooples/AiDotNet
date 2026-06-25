---
title: "FairFeatureSelector<T>"
description: "Fair Feature Selection that balances predictive power with fairness constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Fairness`

Fair Feature Selection that balances predictive power with fairness constraints.

## For Beginners

In ML fairness, we want models that don't discriminate
based on protected attributes like race or gender. This selector chooses features
that are good for prediction but have low correlation with a specified protected
attribute. This helps build fairer models by removing proxies for protected groups.

## How It Works

Selects features that maximize predictive accuracy while minimizing correlation
with protected attributes to promote fair predictions.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDisparateImpact(Matrix<>,Int32)` | Computes the disparate impact ratio for a specific feature. |

