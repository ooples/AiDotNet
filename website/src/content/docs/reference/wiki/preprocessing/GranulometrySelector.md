---
title: "GranulometrySelector<T>"
description: "Granulometry based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Morphological`

Granulometry based Feature Selection.

## For Beginners

Granulometry measures the size distribution of structures
in data by applying progressively larger opening operations. Features with interesting
size distributions (neither too smooth nor too noisy) are selected.

## How It Works

Selects features based on granulometric analysis, measuring how features respond
to morphological openings of increasing size.

