---
title: "RandomProjectionSelector<T>"
description: "Random Projection-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Projection`

Random Projection-based Feature Selection.

## For Beginners

Random projections preserve important structure
in data when projecting to lower dimensions. By looking at which original
features contribute most to these projections, we can identify the most
important features without expensive computations.

## How It Works

Uses random projections to identify features that contribute most to
preserving distances in the projected space (Johnson-Lindenstrauss lemma).

