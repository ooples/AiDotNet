---
title: "MMPCSelector<T>"
description: "MMPC (Max-Min Parents and Children) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Causal`

MMPC (Max-Min Parents and Children) based Feature Selection.

## For Beginners

MMPC finds features that are direct causes or
effects of the target (parents/children in a causal graph). It uses conditional
independence tests to identify these relationships. The result is a minimal
set of truly relevant features, removing spurious correlations.

## How It Works

Selects features using the MMPC algorithm, which identifies the Markov
blanket of the target - the minimal set of features needed for prediction.

**Note:** The alpha parameter in this implementation is used as a minimum association
threshold (correlation magnitude), not as a p-value significance level as in traditional MMPC.
Features with association below this threshold are considered conditionally independent.

