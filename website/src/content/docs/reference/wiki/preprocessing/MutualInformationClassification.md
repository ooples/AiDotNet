---
title: "MutualInformationClassification<T>"
description: "Mutual Information for classification feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Mutual Information for classification feature selection.

## For Beginners

Mutual information measures how much knowing a feature
tells you about the class. If knowing the feature value significantly reduces
uncertainty about the class, the mutual information is high. Unlike correlation,
it can capture any type of relationship, not just linear ones.

## How It Works

Measures the mutual information between each feature and the target class.
Captures any kind of dependency (linear or nonlinear) between features and target.

