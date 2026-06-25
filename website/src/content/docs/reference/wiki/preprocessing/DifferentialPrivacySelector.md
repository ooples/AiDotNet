---
title: "DifferentialPrivacySelector<T>"
description: "Differential Privacy based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Privacy`

Differential Privacy based Feature Selection.

## For Beginners

Differential privacy adds calibrated noise to feature
scores to protect individual privacy. Even if an attacker sees the selected
features, they can't determine if any specific individual was in the dataset.
This uses the exponential mechanism to select features with probability
proportional to their importance while maintaining privacy guarantees.

## How It Works

Selects features using differential privacy mechanisms to protect individual
data points while still identifying important features.

