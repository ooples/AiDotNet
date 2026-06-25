---
title: "ANOVASelector<T>"
description: "ANOVA-based feature selection for multi-class problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical`

ANOVA-based feature selection for multi-class problems.

## For Beginners

ANOVA checks if the average value of a feature is
significantly different across multiple groups. If a feature has very different
averages for each class, it's good at distinguishing between them. Unlike the
t-test, ANOVA works with any number of classes.

## How It Works

Analysis of Variance (ANOVA) extends the t-test to multiple classes. It measures
how much the mean of each feature varies across classes compared to within-class
variation.

