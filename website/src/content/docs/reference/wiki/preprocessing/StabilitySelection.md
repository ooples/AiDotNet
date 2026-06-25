---
title: "StabilitySelection<T>"
description: "Stability Selection for robust feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Stability Selection for robust feature selection.

## For Beginners

A single run of feature selection might pick some features
by chance. Stability Selection runs many rounds on different parts of the data and
counts how often each feature is selected. Features that are picked consistently
(say, 70% of the time) are more trustworthy than those picked rarely.

## How It Works

Stability Selection runs feature selection multiple times on random subsamples
of the data and selects features that are consistently chosen across iterations.
This reduces sensitivity to noise and produces more reliable feature sets.

