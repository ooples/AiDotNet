---
title: "FTest<T>"
description: "F-Test for feature selection in regression problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

F-Test for feature selection in regression problems.

## For Beginners

The F-test asks: "How much of the target's variation
can be explained by this feature compared to random noise?" A high F-value means
the feature explains a significant portion of the target variability. It's used
for continuous targets (regression), not classification.

## How It Works

The F-test computes the ratio of the variance explained by the relationship between
a feature and the target to the unexplained variance. Higher F-statistics indicate
features with stronger linear relationships to the target.

