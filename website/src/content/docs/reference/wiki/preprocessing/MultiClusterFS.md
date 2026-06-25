---
title: "MultiClusterFS<T>"
description: "Multi-Cluster Feature Selection (MCFS) using spectral analysis for unsupervised feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral`

Multi-Cluster Feature Selection (MCFS) using spectral analysis for unsupervised feature selection.

## For Beginners

MCFS first discovers natural groups (clusters) in your data
without knowing the labels. Then it picks features that best explain these groupings.
If a feature helps distinguish between clusters, it's selected. This is useful when
you have no labels but want to find discriminative features.

## How It Works

MCFS uses spectral clustering to discover the underlying cluster structure of the data,
then selects features that best preserve this structure. It combines spectral embedding
with sparse regression to select informative features.

