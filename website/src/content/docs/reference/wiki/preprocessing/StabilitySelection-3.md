---
title: "StabilitySelection<T>"
description: "Stability Selection for robust feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Stability`

Stability Selection for robust feature selection.

## For Beginners

Different samples of your data might give different
feature rankings. Stability selection checks which features are consistently
chosen across many random samples. Features that are always selected are
likely to be truly important, not just lucky in one particular sample.

## How It Works

Stability selection runs feature selection on multiple random subsets of the
data and keeps features that are consistently selected across runs. This
reduces the variance of feature selection and produces more reliable results.

