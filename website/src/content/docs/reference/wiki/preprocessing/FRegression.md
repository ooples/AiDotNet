---
title: "FRegression<T>"
description: "F-statistic based feature selection for regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Regression`

F-statistic based feature selection for regression.

## For Beginners

The F-test checks how well each feature can predict
the target using a simple line (linear relationship). Features that follow
a straighter line with the target get higher scores. Good for finding features
that have clear linear patterns with your outcome.

## How It Works

F-Regression computes the F-statistic between each feature and the target,
which measures the linear dependency between them. Features with higher
F-scores have stronger linear relationships with the target.

