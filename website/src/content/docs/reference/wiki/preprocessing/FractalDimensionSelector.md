---
title: "FractalDimensionSelector<T>"
description: "Fractal Dimension based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Complexity`

Fractal Dimension based Feature Selection.

## For Beginners

Fractal dimension measures how "rough" or complex
a signal is. A straight line has dimension 1, a completely random signal
approaches dimension 2. Complex, irregular patterns have higher fractal
dimensions. This is useful for finding features with interesting structure.

## How It Works

Selects features based on their estimated fractal dimension using
the Higuchi algorithm, measuring signal complexity.

