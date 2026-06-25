---
title: "FClassification<T>"
description: "F-statistic based feature selection for classification (ANOVA F-test)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Classification`

F-statistic based feature selection for classification (ANOVA F-test).

## For Beginners

The F-test checks if the average values of a feature
are different across classes. If class A has very different values than class B
for a feature, that feature gets a high score because it helps tell the classes apart.

## How It Works

F-Classification uses ANOVA F-test to score features based on their ability
to discriminate between class means. Features with high F-scores have
significantly different means across classes.

