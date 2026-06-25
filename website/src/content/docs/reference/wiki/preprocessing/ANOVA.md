---
title: "ANOVA<T>"
description: "ANOVA (Analysis of Variance) F-test for feature selection in classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

ANOVA (Analysis of Variance) F-test for feature selection in classification problems.

## For Beginners

ANOVA asks: "Is the difference between class means large
compared to the scatter within each class?" If class means are very different but
values within each class are similar, the feature is good at separating classes.

## How It Works

ANOVA F-test compares the variance between groups (classes) to the variance within groups.
A high F-statistic indicates that the feature discriminates well between classes.
This is equivalent to f_classif in scikit-learn.

