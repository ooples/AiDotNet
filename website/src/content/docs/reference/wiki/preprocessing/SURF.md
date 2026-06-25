---
title: "SURF<T>"
description: "Spatially Uniform ReliefF (SURF) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Relief`

Spatially Uniform ReliefF (SURF) for feature selection.

## For Beginners

Unlike ReliefF which looks at exactly k neighbors,
SURF looks at all neighbors within a certain distance. This is more natural
because it considers the actual distribution of your data - areas with more
data points contribute more comparisons.

## How It Works

SURF is a variant of ReliefF that uses distance thresholds instead of k
nearest neighbors. It considers all instances within a threshold distance,
making it more robust and parameter-free for the neighbor count.

