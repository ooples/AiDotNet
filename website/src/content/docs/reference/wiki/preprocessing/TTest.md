---
title: "TTest<T>"
description: "Student's t-test for binary classification feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Student's t-test for binary classification feature selection.

## For Beginners

The t-test asks: are the average values of this feature
really different between class 0 and class 1, or could the difference be due to chance?
A small p-value (high t-statistic) means the feature genuinely helps separate classes.

## How It Works

Performs independent two-sample t-tests between feature values for each class.
Features with significant differences between class means are selected as they
help distinguish between classes.

