---
title: "GainRatioSelection<T>"
description: "Gain Ratio Feature Selection (normalized Information Gain)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Gain Ratio Feature Selection (normalized Information Gain).

## For Beginners

Information gain tends to favor features with
many unique values (like IDs), even if they're not truly useful. Gain ratio
fixes this by dividing the information gain by how "spread out" the feature
values are. This gives fairer scores across features with different numbers
of unique values.

## How It Works

Gain Ratio normalizes information gain by the intrinsic information of the
feature (split information). This corrects for the bias of information gain
towards features with many values.

