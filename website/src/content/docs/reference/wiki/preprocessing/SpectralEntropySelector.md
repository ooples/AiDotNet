---
title: "SpectralEntropySelector<T>"
description: "Spectral Entropy based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Spectral`

Spectral Entropy based Feature Selection.

## For Beginners

This analyzes the "frequency content" of each feature's
information. Features with more complex spectral patterns (higher entropy) often
contain more useful information for prediction.

## How It Works

Selects features based on their spectral entropy computed from the eigenvalue
distribution of their covariance structure.

