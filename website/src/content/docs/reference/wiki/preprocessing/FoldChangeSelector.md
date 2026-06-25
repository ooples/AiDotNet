---
title: "FoldChangeSelector<T>"
description: "Fold Change-based feature selection for differential analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bioinformatics`

Fold Change-based feature selection for differential analysis.

## For Beginners

Fold change measures how much a feature's average value
changes between two groups. A fold change of 2 means the feature is twice as high
in one group. Large fold changes indicate features that behave very differently
between conditions.

## How It Works

Selects features based on fold change (ratio of means) between two conditions.
Commonly used in gene expression analysis to identify differentially expressed genes.

