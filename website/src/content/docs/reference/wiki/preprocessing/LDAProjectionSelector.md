---
title: "LDAProjectionSelector<T>"
description: "Linear Discriminant Analysis Projection based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Classification`

Linear Discriminant Analysis Projection based Feature Selection.

## For Beginners

LDA finds directions that best separate classes.
This selector measures how much each feature contributes to these separating
directions, keeping features that help distinguish between groups.

## How It Works

Selects features based on their contribution to class separation using
the ratio of between-class to within-class variance.

