---
title: "ClusterSeparabilitySelector<T>"
description: "Cluster Separability based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Clustering`

Cluster Separability based Feature Selection.

## For Beginners

Good features should make clusters easy to tell apart.
This selector finds features where different groups of data are far from each
other but points within each group are close together.

## How It Works

Selects features based on how well they separate natural clusters in the data,
measuring the ratio of between-cluster to within-cluster variance.

