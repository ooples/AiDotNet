---
title: "BootstrapSelector<T>"
description: "Bootstrap based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.CrossValidation`

Bootstrap based Feature Selection.

## For Beginners

Bootstrap sampling creates many random subsets of
data (with replacement). Features that are consistently important across all
these subsets are more reliable than those that only work on specific subsets.

## How It Works

Selects features based on their importance stability across multiple bootstrap
samples, ensuring robust feature selection.

