---
title: "SPEC<T>"
description: "SPEC (Spectral Feature Selection) for unsupervised feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral`

SPEC (Spectral Feature Selection) for unsupervised feature selection.

## For Beginners

SPEC builds a graph where similar data points are
connected. It then looks at the graph's "shape" (via eigenvectors) and asks:
which features align well with this structure? Features that respect the natural
groupings in the data get high scores.

## How It Works

SPEC ranks features based on their consistency with the structure of the data
as captured by a similarity graph's spectral representation. It uses the
eigenvectors of the graph Laplacian to evaluate feature quality.

