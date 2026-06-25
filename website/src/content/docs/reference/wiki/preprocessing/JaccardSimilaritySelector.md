---
title: "JaccardSimilaritySelector<T>"
description: "Jaccard Similarity based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Similarity`

Jaccard Similarity based Feature Selection.

## For Beginners

Jaccard similarity measures overlap between sets.
After converting features to binary (above/below threshold), it compares
which points are "high" in both feature and target simultaneously.

## How It Works

Selects features based on their Jaccard similarity with the target after
binarization, measuring the intersection over union of positive values.

