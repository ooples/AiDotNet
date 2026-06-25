---
title: "TruncatedSVDSelection<T>"
description: "Truncated SVD Feature Selection using singular value decomposition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Dimensionality`

Truncated SVD Feature Selection using singular value decomposition.

## For Beginners

SVD breaks down your data into components
that explain different directions of variation. By looking at which
original features contribute to the most important directions, we can
select features that capture the most information about your data.

## How It Works

Uses Singular Value Decomposition to identify which original features
contribute most to the principal directions of variation. Unlike PCA,
this works directly on the data matrix without centering.

