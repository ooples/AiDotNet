---
title: "DifferentialEntropySelector<T>"
description: "Differential Entropy based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Entropy`

Differential Entropy based Feature Selection.

## For Beginners

Differential entropy measures uncertainty in
continuous distributions. Unlike discrete entropy, it can be negative.
Features with higher differential entropy contain more unpredictable
information, which may indicate useful variability.

## How It Works

Selects features based on their differential (continuous) entropy,
estimated using nearest-neighbor methods.

