---
title: "ANOVAFTest<T>"
description: "ANOVA F-test for feature selection in multi-class classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

ANOVA F-test for feature selection in multi-class classification.

## For Beginners

ANOVA asks: "Does this feature have different average
values across the different classes?" If a feature has very similar averages
in all classes, it can't help tell them apart. Features with high F-scores
have means that vary significantly between classes.

## How It Works

Uses one-way Analysis of Variance to test if feature means differ significantly
across multiple classes. Features with high F-statistics indicate strong
discriminative power between classes.

