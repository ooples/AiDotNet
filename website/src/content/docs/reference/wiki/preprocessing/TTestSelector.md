---
title: "TTestSelector<T>"
description: "T-Test based feature selection for binary classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical`

T-Test based feature selection for binary classification.

## For Beginners

The t-test asks "are the average values of this
feature significantly different between the two groups?" A high t-statistic
means the feature's values are very different for each class, making it useful
for telling them apart.

## How It Works

Uses independent samples t-test to measure how well each feature separates
two classes. Features with higher t-statistics have significantly different
means between classes.

